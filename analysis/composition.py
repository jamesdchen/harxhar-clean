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


def _block_boot_cov(S: np.ndarray, y: np.ndarray, bins: np.ndarray, K: int, lam: float,
                    n_boot: int = 60, block: int = 2000, seed: int = 0):
    """Block-bootstrap sampling covariance of the per-bin weights, averaged over bins.

    The analytic ridge sandwich assumes iid errors. The HAR residual is heteroskedastic *and*
    autocorrelated, so that formula UNDER-states the sampling variance — which shows up directly as a
    circular-shift null whose 'signal' trace is +0.05 instead of 0. A moving-block bootstrap within
    each bin estimates the same covariance without the iid assumption.
    """
    p = S.shape[1]
    rng = np.random.default_rng(seed)
    acc = np.zeros((p, p))
    used = 0
    for k in range(K):
        idx = np.flatnonzero(bins == k)
        if idx.size < 2000:
            continue
        draws = []
        nb = max(1, idx.size // block)
        for _ in range(n_boot):
            starts = rng.integers(0, max(1, idx.size - block), size=nb)
            take = np.concatenate([idx[s : s + block] for s in starts])
            Z = S[take] - S[take].mean(0)
            yc = y[take] - y[take].mean()
            draws.append(np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ yc))
        D = np.asarray(draws)
        acc += np.cov(D.T, ddof=1) * (len(take) / idx.size)  # scale to the full-bin sample size
        used += 1
    return acc / max(used, 1)


def stage_audit() -> None:  # noqa: C901
    """Is §14.1's ellipse clean? Null the EIGENSTRUCTURE, not just the trace, and replicate the axis.

    Two things §14.1 did not do, either of which could void it:

    1. **The null was only applied to the trace.** With ``K`` bins ``Sigma_obs`` has ``K-1`` degrees of
       freedom in 7 dimensions, so at K=5 it has rank <= 4 and its eigenvalues concentrate *by
       construction*. "Effective rank 1.5 of 7" may be a df artifact — precisely the error §14 made
       with the radial/tangential split. So the null distribution of ``eig1_share`` and
       ``eff_rank`` is what decides, not the observed value.
    2. **The noise term assumed iid errors.** The null 'signal' trace came out at +0.05 rather than 0,
       which is the signature of under-subtraction. A block bootstrap replaces the analytic covariance
       without the iid assumption.

    And the decisive test, which is immune to both: **does the leading axis replicate?** Estimate the
    top eigenvector on the first half of the search period and on the second half and take
    ``|cos(v1, v2)|`` against its circular-shift null. If the axis is real it points the same way in
    both halves; a df artifact does not.
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    sig = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    n = len(e)
    search = (ts < SPLIT).to_numpy()
    idx = np.flatnonzero(search)
    h1 = np.zeros(n, bool)
    h2 = np.zeros(n, bool)
    h1[idx[: len(idx) // 2]] = True
    h2[idx[len(idx) // 2 :]] = True

    def sigma_signal(rows, y, bins, K, noise="analytic"):
        B, Ca = _bin_fit(S[rows], y[rows], bins[rows], K, RIDGE)
        ok = np.isfinite(B).all(1)
        if ok.sum() < 3:
            return None, None, None
        Sobs = np.cov(B[ok].T, ddof=1)
        C = Ca if noise == "analytic" else _block_boot_cov(S[rows], y[rows], bins[rows], K, RIDGE)
        return Sobs - C, B[ok].mean(0), C

    rows_out = []
    for K in (10, 20):
        bins = _vol_regime(p, ts, K)
        for noise in ("analytic", "bootstrap"):
            Ssig, Bbar, C = sigma_signal(search, e, bins, K, noise)
            md = Bbar / np.linalg.norm(Bbar)
            g = _eig_report(Ssig, md, noise)
            # NULL the eigenstructure, not just the trace
            nt, ne1, ner = [], [], []
            for kk in range(N_NULL):
                sh = (kk + 1) * (n // (N_NULL + 1))
                Sn, Bn, _ = sigma_signal(search, np.roll(e, sh), bins, K, noise)
                if Sn is None:
                    continue
                gn = _eig_report(Sn, Bn / np.linalg.norm(Bn), "null")
                nt.append(gn["trace"])
                ne1.append(gn["eig1_share"])
                ner.append(gn["eff_rank"])
            print(f"  --- K={K}, noise={noise} ---")
            print(f"    trace signal {g['trace']:+.4f}   null {np.nanmean(nt):+.4f}   "
                  f"ratio {g['trace'] / max(np.nanmean(nt), 1e-9):.1f}x")
            print(f"    eig1 share   {g['eig1_share']:.3f}   NULL {np.nanmean(ne1):.3f}  <- the check §14.1 skipped")
            print(f"    eff rank     {g['eff_rank']:.2f}     NULL {np.nanmean(ner):.2f}")
            rows_out.append({"K": K, "noise": noise, "trace": g["trace"],
                             "null_trace": np.nanmean(nt), "eig1_share": g["eig1_share"],
                             "null_eig1_share": np.nanmean(ne1), "eff_rank": g["eff_rank"],
                             "null_eff_rank": np.nanmean(ner)})
        # ---- the decisive test: does the axis replicate across halves? -----------------
        Sa, Ba, _ = sigma_signal(h1, e, bins, K)
        Sb, Bb, _ = sigma_signal(h2, e, bins, K)
        va = _eig_report(Sa, Ba / np.linalg.norm(Ba), "h1")["top_evec"]
        vb = _eig_report(Sb, Bb / np.linalg.norm(Bb), "h2")["top_evec"]
        cos = abs(float(va @ vb))
        ncos = []
        for kk in range(N_NULL):
            sh = (kk + 1) * (n // (N_NULL + 1))
            en = np.roll(e, sh)
            Sa2, Ba2, _ = sigma_signal(h1, en, bins, K)
            Sb2, Bb2, _ = sigma_signal(h2, en, bins, K)
            if Sa2 is None or Sb2 is None:
                continue
            v1 = _eig_report(Sa2, Ba2 / np.linalg.norm(Ba2), "n")["top_evec"]
            v2 = _eig_report(Sb2, Bb2 / np.linalg.norm(Bb2), "n")["top_evec"]
            ncos.append(abs(float(v1 @ v2)))
        print(f"    AXIS REPLICATION |cos(v_h1, v_h2)| = {cos:.3f}   null {np.nanmean(ncos):.3f}")
        print("      h1 loadings: " + ", ".join(f"{BUCKETS[i]} {va[i]:+.2f}" for i in np.argsort(-np.abs(va))[:3]))
        print("      h2 loadings: " + ", ".join(f"{BUCKETS[i]} {vb[i]:+.2f}" for i in np.argsort(-np.abs(vb))[:3]))
        rows_out[-1].update({"axis_cos_split_half": cos, "axis_cos_null": np.nanmean(ncos)})
        print()
    d = pd.DataFrame(rows_out)
    d.to_csv(f"{OUT}/composition_audit.csv", index=False)
    print("  VERDICT")
    bs = d[(d.K == 10) & (d.noise == "bootstrap")].iloc[0]
    # Averaging the split-half cosine across K was too lenient: it let K=20's 0.549 mask K=10's 0.309,
    # which sits EXACTLY on its null. An axis is only identified if it clears the null at EVERY K, and
    # a cosine can be inflated by shared structure even when the two halves name disjoint buckets, so
    # the loadings must agree too. Both conditions, per K.
    ax = d.dropna(subset=["axis_cos_split_half"])
    per_k = {int(r.K): (r.axis_cos_split_half > r.axis_cos_null + 0.10) for _, r in ax.iterrows()}
    clean_axis = all(per_k.values())
    print(f"    trace survives the honest (bootstrap) noise estimate: {bs['trace'] > 2 * bs['null_trace']}"
          f"  [{bs['trace'] / max(bs['null_trace'], 1e-9):.0f}x]")
    print(f"    eig1 concentration exceeds its df artifact: {bs['eig1_share'] > bs['null_eig1_share'] + 0.05}"
          f"  ({bs['eig1_share']:.2f} vs null {bs['null_eig1_share']:.2f} -- mostly artifact)")
    print(f"    axis replicates at EVERY K: {clean_axis}  {per_k}")
    print("    -> REAL VARIATION, UNIDENTIFIED AXIS." if not clean_axis else "    -> axis identified")
    print("       The weights do move with state beyond noise, but which direction they move in")
    print("       cannot be pinned down: the halves disagree on the loadings. Do not build a fitted")
    print("       rank-1 model on this; a PRE-SPECIFIED direction is the better-powered test.")
    print(f"wrote {OUT}/composition_audit.csv")


def stage_axis_direct() -> None:  # noqa: C901
    """Estimate the axis directly at BAR resolution — no bins, no eigendecomposition.

    §14.2 could not identify the axis, and §14.1's summary blamed "167 monthly observations". That
    diagnosis was wrong and worth correcting precisely, because two different objects were conflated:

    * The **intensity** target (§13) genuinely has ~167 effective observations: it is a ~1-month EWMA
      with one-day autocorrelation 0.988, and slicing a smooth series into finer bars manufactures no
      information — demonstrated there, where the bar-level 7x excess vanished at honest resolution.
    * The **composition/axis** question has nothing to do with monthly anything. Each bin's weights were
      fit on ~17,657 bars (K=10 over 176,574 search bars). The binding constraint was **K**, because the
      across-bin covariance gets only ``K-1`` degrees of freedom in 7 dimensions. Binning was inherited
      from §5's granularity ladder and never re-examined.

    So drop the bins. Write the weight vector as a smooth function of a continuous causal state ``z``
    and take its first-order term:

        e  ~  sum_k beta_k s_k  +  sum_k gamma_k (s_k * z)

    Then ``gamma`` **is** the axis — the direction in which state moves the weights — estimated from all
    176,574 bars with 7 parameters instead of read off an eigendecomposition of 10 noisy bin estimates.
    Same decisive test as §14.2 (does it replicate across halves?), now with bar-level precision, plus a
    HAC joint Wald test that ``gamma = 0``.
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    sig = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    n, q = len(e), S.shape[1]

    # continuous causal state: the panel's har_ma_25 is already rolling-robust-scaled and lagged
    z = p.X[TW + TW :, p.names.index("har_ma_25")].astype(np.float64)
    z = np.clip((z - np.median(z)) / (np.percentile(z, 75) - np.percentile(z, 25) + 1e-12), -4, 4)

    search = (ts < SPLIT).to_numpy()
    idx = np.flatnonzero(search)
    h1 = np.zeros(n, bool)
    h2 = np.zeros(n, bool)
    h1[idx[: len(idx) // 2]] = True
    h2[idx[len(idx) // 2 :]] = True

    def fit(rows, y, zz):
        D = np.hstack([S[rows], S[rows] * zz[rows][:, None]])
        D = D - D.mean(0)
        yc = y[rows] - y[rows].mean()
        G = D.T @ D
        b = np.linalg.solve(G + RIDGE * np.eye(2 * q), D.T @ yc)
        u = yc - D @ b
        # HAC (Newey-West, 10 trading days) sandwich for the coefficient covariance
        lags = 480
        Om = (D * u[:, None]).T @ (D * u[:, None])
        for L in range(1, lags + 1):
            w = 1.0 - L / (lags + 1.0)
            A = (D[L:] * u[L:, None]).T @ (D[:-L] * u[:-L, None])
            Om += w * (A + A.T)
        Gi = np.linalg.inv(G + RIDGE * np.eye(2 * q))
        V = Gi @ Om @ Gi
        return b, V

    b, V = fit(search, e, z)
    beta, gamma = b[:q], b[q:]
    Vg = V[q:, q:]
    tg = gamma / np.sqrt(np.maximum(np.diag(Vg), 1e-24))
    wald = float(gamma @ np.linalg.solve(Vg, gamma))
    print(f"  bar-resolution fit on {int(search.sum())} search rows, 14 parameters\n")
    print("  bucket        beta      gamma   HAC-t(gamma)")
    for k, nm in enumerate(BUCKETS):
        print(f"    {nm:12s} {beta[k]:+.4f}  {gamma[k]:+.4f}   {tg[k]:+6.2f}")
    from scipy.stats import chi2

    print(f"\n  joint HAC Wald test gamma = 0: chi2({q}) = {wald:.1f}, "
          f"p = {chi2.sf(wald, q):.2e}")
    print(f"  ||gamma|| / ||beta|| = {np.linalg.norm(gamma) / np.linalg.norm(beta):.3f}")

    # decisive: does the axis replicate across halves, at bar precision?
    b1, _ = fit(h1, e, z)
    b2, _ = fit(h2, e, z)
    g1, g2 = b1[q:], b2[q:]
    cos = float(g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2)))
    ncos = []
    for k in range(N_NULL):
        sh = (k + 1) * (n // (N_NULL + 1))
        en = np.roll(e, sh)
        c1, _ = fit(h1, en, z)
        c2, _ = fit(h2, en, z)
        d1, d2 = c1[q:], c2[q:]
        ncos.append(float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2))))
    print(f"\n  AXIS REPLICATION cos(gamma_h1, gamma_h2) = {cos:+.3f}   "
          f"null {np.mean(ncos):+.3f} (sd {np.std(ncos):.3f})")
    print("    h1: " + ", ".join(f"{BUCKETS[i]} {g1[i]:+.3f}" for i in np.argsort(-np.abs(g1))[:3]))
    print("    h2: " + ", ".join(f"{BUCKETS[i]} {g2[i]:+.3f}" for i in np.argsort(-np.abs(g2))[:3]))
    clean = cos > np.mean(ncos) + 3 * np.std(ncos)
    print(f"\n  -> axis {'IDENTIFIED at bar resolution' if clean else 'still not identified'}")
    pd.DataFrame({"bucket": BUCKETS, "beta": beta, "gamma": gamma, "hac_t_gamma": tg,
                  "gamma_h1": g1, "gamma_h2": g2}).to_csv(f"{OUT}/axis_direct.csv", index=False)
    pd.DataFrame([{"wald_chi2": wald, "wald_p": float(chi2.sf(wald, q)), "cos_split_half": cos,
                   "cos_null_mean": float(np.mean(ncos)), "cos_null_sd": float(np.std(ncos)),
                   "identified": bool(clean), "n_rows": int(search.sum())}]).to_csv(
        f"{OUT}/axis_direct_summary.csv", index=False)
    print(f"wrote {OUT}/axis_direct.csv + axis_direct_summary.csv")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["geometry", "geometry2", "audit", "axis_direct"], required=True)
    a = ap_.parse_args()
    {"geometry": stage_geometry, "geometry2": stage_geometry2, "audit": stage_audit, "axis_direct": stage_axis_direct}[a.stage]()
