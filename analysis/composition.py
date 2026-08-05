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


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["geometry"], required=True)
    a = ap_.parse_args()
    {"geometry": stage_geometry}[a.stage]()
