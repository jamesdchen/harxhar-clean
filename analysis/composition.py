"""Is the state-dependence a rotation of composition at fixed magnitude — a partition of unity?

§13.1 concluded that the dense-weak alpha's state-dependence is **not** a modulation of magnitude but
a **re-allocation across buckets**: a state-dependent gain on the whole composite loses at every
granularity (§5's 1-df-per-bin gain, -0.0006 at K=2), while re-weighting *among* the 7 buckets pays
(§11.1, RW adjusted p 0.0005). That implies a decomposition

    beta(s) = rho * u(s),      ||u(s)|| = 1

with the magnitude ``rho`` roughly state-invariant and the unit direction ``u(s)`` rotating. This
module measures whether that is actually true, and if so **which** geometry holds — because the answer
picks the estimator:

* **simplex / partition of unity** — weights non-negative and ``sum_k beta_k(s)`` constant. Then the
  natural model is a *mixture of experts with a softmax gate*: the gate is literally a partition of
  unity over the state space, the experts are the bucket signals, and liquidity "owning" the calm
  patch while moments "owns" the stressed patch (§4) is exactly local support with smooth blending.
* **sphere** — ``||beta(s)||_2`` constant but signs free. Then the model is a rotation: one base
  direction plus a small number of rotation generators.
* **neither** — magnitude also moves, and §13.1's reframing is wrong.

The discriminating measurement is a **radial / tangential variance decomposition** of the per-bin
weight vectors. Write each bin's fitted ``beta_b`` in polar form; the variation across bins splits into
a *radial* part (magnitude changing) and a *tangential* part (direction rotating at fixed magnitude).
§13.1 predicts tangential >> radial. Everything here is descriptive on the **search period only**
(<= 2020) against circular-shift nulls, so it scores no forecast and consumes no inferential budget —
which matters because the 2021-2024 holdout has already been evaluated twice (§12, §12.1).

Usage
-----
    python analysis/composition.py --stage geometry
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import BUCKETS, TW, _vol_regime
from analysis.alpha_panel import CACHE_DIR

RIDGE = 1.0  # the §5 combiner's penalty on 7 standardized signals
SPLIT = "2021-01-01"
N_NULL = 24
OUT = "results/alpha_manifestation"
FIXED_CACHE = os.path.join(CACHE_DIR, "fixed")


def _panel():
    os.environ["ALPHA_PANEL_CACHE"] = FIXED_CACHE
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    return ap.load_panel()


def _bin_weights(S: np.ndarray, y: np.ndarray, bins: np.ndarray, K: int) -> np.ndarray:
    """(K, 7) ridge weight vector fitted within each bin."""
    out = np.full((K, S.shape[1]), np.nan)
    for k in range(K):
        m = bins == k
        if m.sum() < 500:
            continue
        Z = S[m] - S[m].mean(0)
        yc = y[m] - y[m].mean()
        out[k] = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(S.shape[1]), Z.T @ yc)
    return out


def _geometry(B: np.ndarray) -> dict:
    """Radial vs tangential decomposition of the across-bin variation in weight vectors.

    ``B`` is (K, 7). Let ``r_b = ||B_b||`` and ``u_b = B_b / r_b``. Total across-bin variation is
    ``sum_b ||B_b - Bbar||^2``; the radial part is what a *pure magnitude* change would explain
    (projections on the mean direction), and the tangential remainder is genuine rotation. A
    partition-of-unity / simplex structure additionally requires non-negative entries and a constant
    ``sum_k B_bk``.
    """
    ok = np.isfinite(B).all(1)
    B = B[ok]
    r = np.linalg.norm(B, axis=1)
    U = B / r[:, None]
    ubar = U.mean(0)
    ubar /= np.linalg.norm(ubar)
    Bbar = B.mean(0)
    total = float(((B - Bbar) ** 2).sum())
    # radial: the component of each deviation along the mean direction
    radial = float((((B - Bbar) @ ubar) ** 2).sum())
    tangential = max(total - radial, 0.0)
    cos = U @ ubar
    return {
        "n_bins": int(ok.sum()),
        "l2_mean": float(r.mean()),
        "l2_cv": float(r.std() / r.mean()) if r.mean() else np.nan,
        "l1_cv": float(np.abs(B).sum(1).std() / np.abs(B).sum(1).mean()),
        "sum_cv": float(B.sum(1).std() / abs(B.sum(1).mean())) if B.sum(1).mean() else np.nan,
        "neg_share": float((B < 0).mean()),
        "radial_share": radial / total if total > 0 else np.nan,
        "tangential_share": tangential / total if total > 0 else np.nan,
        "min_cos_to_mean": float(cos.min()),
        "max_angle_deg": float(np.degrees(np.arccos(np.clip(cos.min(), -1, 1)))),
    }


def stage_geometry() -> None:
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    sig = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    n = len(e)
    search = (ts < SPLIT).to_numpy()
    print(f"  search rows {int(search.sum())} of {n}; 7 bucket signals\n")

    rows = []
    for K in (2, 3, 5):
        bins = _vol_regime(p, ts, K)
        B = _bin_weights(S[search], e[search], bins[search], K)
        g = _geometry(B)
        # null: circular-shift the residual, refit, and redo the decomposition
        nt, nr = [], []
        for k in range(N_NULL):
            sh = (k + 1) * (n // (N_NULL + 1))
            en = np.roll(e, sh)
            gn = _geometry(_bin_weights(S[search], en[search], bins[search], K))
            nt.append(gn["tangential_share"])
            nr.append(gn["l2_cv"])
        print(f"  --- vol regime, K={K} ---")
        print(f"    ||beta||_2 across bins: mean {g['l2_mean']:.4f}  CV {g['l2_cv']:.3f}  "
              f"(null CV {np.nanmean(nr):.3f})")
        print(f"    CV of L1 norm {g['l1_cv']:.3f} | CV of the plain SUM {g['sum_cv']:.3f} "
              f"| negative-weight share {g['neg_share']:.2f}")
        print(f"    variance split: radial {g['radial_share']:.2f}  tangential "
              f"{g['tangential_share']:.2f}  (null tangential {np.nanmean(nt):.2f})")
        print(f"    max rotation from the mean direction: {g['max_angle_deg']:.1f} deg\n")
        g.update({"K": K, "null_tangential": np.nanmean(nt), "null_l2_cv": np.nanmean(nr)})
        rows.append(g)
        pd.DataFrame(B, columns=BUCKETS).to_csv(f"{OUT}/composition_weights_K{K}.csv", index=False)

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/composition_geometry.csv", index=False)
    # verdict
    best = d.loc[d.K == 5].iloc[0]
    print("  VERDICT")
    if best["neg_share"] > 0.1:
        print(f"    weights are NOT non-negative ({best['neg_share']:.0%} negative) -> not a simplex,")
        print("    so not a partition of unity in the strict sense.")
    which = min(
        [("L2", best["l2_cv"]), ("L1", best["l1_cv"]), ("sum", abs(best["sum_cv"]))],
        key=lambda t: t[1],
    )
    print(f"    most nearly conserved quantity: {which[0]} (CV {which[1]:.3f})")
    print(f"    rotation vs rescaling: tangential {best['tangential_share']:.2f} vs radial "
          f"{best['radial_share']:.2f}")
    print(f"wrote {OUT}/composition_geometry.csv + composition_weights_K*.csv")


def _bin_fit(S: np.ndarray, y: np.ndarray, bins: np.ndarray, K: int, lam: float):
    """Per-bin ridge weights AND their analytic sampling covariance.

    For ridge, ``A = (Z'Z + lam I)^-1`` and ``Cov(beta_hat) = sigma^2 A (Z'Z) A`` (the sandwich).
    Having this in closed form is what makes the geometry question answerable: the observed
    across-bin scatter is ``Sigma_signal + E[Cov(beta_hat)]``, so the noise term can be *subtracted*
    rather than merely shrunk.
    """
    p = S.shape[1]
    B = np.full((K, p), np.nan)
    C = np.zeros((p, p))
    used = 0
    for k in range(K):
        m = bins == k
        if m.sum() < 500:
            continue
        Z = S[m] - S[m].mean(0)
        yc = y[m] - y[m].mean()
        G = Z.T @ Z
        A = np.linalg.inv(G + lam * np.eye(p))
        b = A @ Z.T @ yc
        B[k] = b
        s2 = float(((yc - Z @ b) ** 2).sum() / max(m.sum() - p, 1))
        C += s2 * A @ G @ A
        used += 1
    return B, C / max(used, 1)


def _eig_report(Sig: np.ndarray, mean_dir: np.ndarray, tag: str) -> dict:
    """Eigen-geometry of a (possibly noise-corrected) across-bin covariance."""
    ev, V = np.linalg.eigh(Sig)
    order = np.argsort(-ev)
    ev, V = ev[order], V[:, order]
    pos = ev[ev > 0]
    tot = float(pos.sum())
    radial = float(mean_dir @ Sig @ mean_dir)  # variance along the mean direction
    return {
        "tag": tag,
        "trace": float(np.trace(Sig)),
        "n_positive_eig": int((ev > 0).sum()),
        "n_negative_eig": int((ev < -1e-12).sum()),
        "eig1_share": float(pos[0] / tot) if tot > 0 and pos.size else np.nan,
        "eig2_share": float(pos[1] / tot) if tot > 0 and pos.size > 1 else np.nan,
        "eff_rank": float(pos.sum() ** 2 / (pos**2).sum()) if pos.size else np.nan,
        "radial_share": radial / float(np.trace(Sig)) if np.trace(Sig) > 0 else np.nan,
        "top_evec": V[:, 0],
    }


def stage_geometry2() -> None:  # noqa: C901
    """Redo §14's decomposition on shrunk weights, then do it properly by subtracting the noise.

    Two things, in order.

    **1. Shrinkage does not answer the question.** §14's radial/tangential shares are computed on
    *deviations from the mean* weight vector. Shrinking each bin toward the pooled vector (the 50/50
    blend that §11.1 showed actually pays) multiplies every deviation by the same factor, so the
    shares are **exactly invariant** to it, and a heavier ridge only rescales them slightly. Shown
    explicitly across a penalty ladder rather than asserted.

    **2. The right fix is to subtract the estimation-error covariance, which is known.** The observed
    across-bin scatter is ``Sigma_obs = Sigma_signal + E[Cov(beta_hat)]``, and the second term is
    available in closed form from the ridge sandwich (:func:`_bin_fit`). So form
    ``Sigma_signal = Sigma_obs - mean(Cov(beta_hat))`` and read its eigen-geometry:

    * **not positive semi-definite / trace <= 0** -> the across-bin variation is *entirely* estimation
      noise. No geometry, nothing to exploit.
    * **one dominant eigenvalue** -> a single axis of state-dependent variation: an *ellipse*, and the
      leading eigenvector says which bucket combination moves. This is the exploitable case, and it is
      cheap: one extra parameter, not K x 7.
    * **near-isotropic (effective rank ~ 7)** -> real variation with no low-dimensional structure, so
      nothing beyond per-bin fitting, which §11.1 already showed is too noisy to pay unshrunk.

    Validation: under a circular-shift null the corrected covariance must collapse (trace ~ 0 or
    indefinite). If it does not, the noise correction is wrong and the rest is void. More bins are used
    than §14 (K up to 20) because ``Sigma_obs`` needs degrees of freedom, and the noise term is now
    subtracted rather than feared.
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    sig = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    n = len(e)
    search = (ts < SPLIT).to_numpy()
    Ss, es = S[search], e[search]

    print("  1. SHRINKAGE LADDER — shares are invariant to shrinking toward the pooled vector\n")
    rows = []
    for lam in (1.0, 1e2, 1e4, 1e6):
        bins = _vol_regime(p, ts, 5)[search]
        B, _ = _bin_fit(Ss, es, bins, 5, lam)
        g = _geometry(B)
        # the 50/50 blend that §11.1 showed pays
        Bb = 0.5 * B + 0.5 * np.nanmean(B, axis=0)
        gb = _geometry(Bb)
        print(f"    ridge {lam:8.0e}: raw tangential {g['tangential_share']:.3f} | "
              f"50/50-blended tangential {gb['tangential_share']:.3f} | "
              f"||beta||_2 CV raw {g['l2_cv']:.3f} blended {gb['l2_cv']:.3f}")
        rows.append({"lam": lam, "tangential_raw": g["tangential_share"],
                     "tangential_blend": gb["tangential_share"], "l2_cv_raw": g["l2_cv"],
                     "l2_cv_blend": gb["l2_cv"]})
    pd.DataFrame(rows).to_csv(f"{OUT}/composition_shrinkage_ladder.csv", index=False)

    print("\n  2. NOISE-CORRECTED COVARIANCE — Sigma_signal = Sigma_obs - E[Cov(beta_hat)]\n")
    out = []
    for K in (5, 10, 20):
        bins_full = _vol_regime(p, ts, K)
        bins = bins_full[search]
        B, Cerr = _bin_fit(Ss, es, bins, K, RIDGE)
        ok = np.isfinite(B).all(1)
        Bk = B[ok]
        Bbar = Bk.mean(0)
        Sobs = np.cov(Bk.T, ddof=1)
        Ssig = Sobs - Cerr
        md = Bbar / np.linalg.norm(Bbar)
        go, gs = _eig_report(Sobs, md, "observed"), _eig_report(Ssig, md, "signal")
        # null validation
        tr_null = []
        for kk in range(N_NULL):
            sh = (kk + 1) * (n // (N_NULL + 1))
            en = np.roll(e, sh)[search]
            Bn, Cn = _bin_fit(Ss, en, bins, K, RIDGE)
            on = np.isfinite(Bn).all(1)
            tr_null.append(float(np.trace(np.cov(Bn[on].T, ddof=1) - Cn)))
        print(f"    --- K={K} bins ({int(ok.sum())} usable) ---")
        print(f"      trace: observed {go['trace']:.5f}  noise {np.trace(Cerr):.5f}  "
              f"SIGNAL {gs['trace']:.5f}  (null signal trace {np.mean(tr_null):+.5f})")
        print(f"      signal eigenvalues: {gs['n_positive_eig']} positive / "
              f"{gs['n_negative_eig']} negative; eig1 share {gs['eig1_share']:.2f}, "
              f"eig2 {gs['eig2_share']:.2f}, effective rank {gs['eff_rank']:.2f} of 7")
        print(f"      radial share of signal {gs['radial_share']:.2f} (isotropic baseline {1/7:.2f})")
        if gs["trace"] > 0 and np.isfinite(gs["eig1_share"]):
            v = gs["top_evec"]
            top = np.argsort(-np.abs(v))
            print("      leading signal direction: " + ", ".join(
                f"{BUCKETS[i]} {v[i]:+.2f}" for i in top[:4]))
        rec = {k: val for k, val in gs.items() if k != "top_evec"}
        rec.update({"K": K, "trace_observed": go["trace"], "trace_noise": float(np.trace(Cerr)),
                    "null_signal_trace": float(np.mean(tr_null))})
        out.append(rec)
        print()
    d = pd.DataFrame(out)
    d.to_csv(f"{OUT}/composition_geometry2.csv", index=False)
    print("  VERDICT")
    best = d.loc[d.K == 10].iloc[0]
    if best["trace"] <= 0 or best["trace"] < 2 * abs(best["null_signal_trace"]):
        print("    signal covariance does not survive the noise subtraction -> NO exploitable geometry")
    elif best["eig1_share"] > 0.6:
        print(f"    one dominant axis (eig1 {best['eig1_share']:.2f}) -> an ELLIPSE: state moves the")
        print("    weights along essentially one direction, worth 1 extra parameter")
    else:
        print(f"    effective rank {best['eff_rank']:.1f} of 7 -> real but near-isotropic variation;")
        print("    no low-dimensional structure to exploit beyond per-bin fitting")
    print(f"wrote {OUT}/composition_geometry2.csv + composition_shrinkage_ladder.csv")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["geometry", "geometry2"], required=True)
    a = ap_.parse_args()
    {"geometry": stage_geometry, "geometry2": stage_geometry2}[a.stage]()
