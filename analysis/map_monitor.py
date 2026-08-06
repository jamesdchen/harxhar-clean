"""The interaction map as a rot detector, plus the flow structure of the factor trajectory.

Two instruments, both built on the study's one replicating geometric object (the factor-pair IC
map, split-half +0.62 vs null 0.00 ± 0.11) and its frame (frozen first-window PCA):

``calib``
    The rot detector's calibration trace. Freeze the frame and a REFERENCE map (first five years of
    the OOS span), then walk a trailing two-year map quarterly through history and record its
    correlation with the reference. The historical distribution of that correlation is what
    anchors the warn/fail thresholds wired into ``src.diagnostics.interaction_map_health`` — the
    §16 philosophy: thresholds from measured levels, not aesthetics. A production run recomputes
    the trailing map quarterly; decorrelation below the historical operating range means the
    interaction channel's support is rotting and the frozen product/pool block should be re-drawn
    (the §"reselection" answer: identity is re-estimated on structural break, not on a calendar).

``leadlag``
    The Cucuringu connection, aimed at the object that finally deserves it. §17.4 dismissed
    signed-graph methods because the block partition did not forecast; §23 changed the target: the
    stable objects are now the FRAME and the trajectory on it, and the trajectory's directed
    structure — which factors lead which at a one-day lag — is exactly the lead-lag network his
    Hermitian-clustering program is built for. Construct the antisymmetric part of the lag-1-day
    cross-correlation among the 20 factor scores (the Hermitian embedding's raw material), test
    its split-half replication like every other object in this study, and read off the flow:
    which factors are sources (net leaders), which are sinks, and where the §23 signal carriers
    sit in that ordering. Descriptive; no forecast scored.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.nl_sparsity import base_columns  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

OUT = "results/alpha_manifestation"
QPOOL = 20
REF_YEARS = 5
TRAIL_YEARS = 2
STEP_MONTHS = 3


def _frame_and_scores():
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    mu0, sd0 = X[:TW].mean(0), X[:TW].std(0)
    live = sd0 > 1e-8
    sd0 = np.where(live, sd0, 1.0)
    lam_l, V_l = np.linalg.eigh(np.corrcoef(((X[:TW] - mu0) / sd0)[:, live], rowvar=False))
    order = np.argsort(lam_l)[::-1]
    V = np.zeros((X.shape[1], len(lam_l)))
    V[live] = V_l[:, order]
    W = V[:, :QPOOL] / sd0[:, None]
    G = (X[TW:] - X[TW:].mean(0)) @ W
    G = (G - G.mean(0)) / (G.std(0) + 1e-12)
    e = e_full[TW:]
    ts = pd.Series(pd.to_datetime(p.t[2 * TW :]))
    return G, e, ts


def _map(Q: np.ndarray, e: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ec = e[mask] - e[mask].mean()
    ec /= ec.std() + 1e-12
    Qm = Q[mask]
    Qm = (Qm - Qm.mean(0)) / (Qm.std(0) + 1e-12)
    return (Qm * ec[:, None]).mean(0)


def stage_calib() -> None:
    os.makedirs(OUT, exist_ok=True)
    G, e, ts = _frame_and_scores()
    qi, qj = np.triu_indices(QPOOL)
    Q = G[:, qi] * G[:, qj]
    n = len(e)
    ref_len = REF_YEARS * 252 * PERIODS_PER_DAY
    trail = TRAIL_YEARS * 252 * PERIODS_PER_DAY
    step = STEP_MONTHS * 21 * PERIODS_PER_DAY
    ref = _map(Q, e, np.arange(n) < ref_len)
    rows = []
    for end in range(ref_len + trail, n, step):
        mask = np.zeros(n, bool)
        mask[end - trail : end] = True
        m = _map(Q, e, mask)
        rows.append({"asof": str(ts.iloc[end - 1].date()),
                     "corr_to_reference": float(np.corrcoef(m, ref)[0, 1])})
    d = pd.DataFrame(rows)
    c = d["corr_to_reference"]
    print(d.to_string(index=False))
    print(f"\ntrailing-2y map vs frozen 5y reference, quarterly {len(d)} points:")
    print(f"  mean {c.mean():+.3f}  min {c.min():+.3f}  p10 {c.quantile(0.1):+.3f}  "
          f"max {c.max():+.3f}")
    print("  (shift-null for a map correlation at this sample: 0.00 +/- 0.11)")
    d.to_csv(f"{OUT}/map_rot_calibration.csv", index=False)
    print(f"wrote {OUT}/map_rot_calibration.csv")


def stage_leadlag() -> None:
    os.makedirs(OUT, exist_ok=True)
    G, e, ts = _frame_and_scores()
    day_last = np.flatnonzero(ts.dt.date.ne(ts.dt.date.shift(-1)).to_numpy())
    Gd = G[day_last]
    nd = len(Gd)

    def antisym(g: np.ndarray) -> np.ndarray:
        a, b = g[:-1], g[1:]
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
        C1 = (a.T @ b) / len(a)  # C1[i,j] = corr(G_i today, G_j tomorrow): i leads j
        return (C1 - C1.T) / 2.0

    A = antisym(Gd)
    h = nd // 2
    A1, A2 = antisym(Gd[:h]), antisym(Gd[h:])
    iu = np.triu_indices(QPOOL, k=1)
    stab = float(np.corrcoef(A1[iu], A2[iu])[0, 1])
    print(f"lead-lag antisymmetric flow: split-half corr {stab:+.3f} over {len(iu[0])} edges "
          f"({nd} days)")
    lead = A.sum(1)  # net leadingness: positive = source, negative = sink
    energy = np.array([4.87, 9.47, 1.77, 14.32, 1.20, 4.42, 0.82, 1.29, 3.83, 10.56,
                       4.37, 10.13, 1.31, 0.93, 8.11, 2.88, 10.88, 2.48, 0.54, 8.57])
    print(f"corr(net leadingness, signal energy): "
          f"{np.corrcoef(lead, energy)[0, 1]:+.3f}   rank "
          f"{np.corrcoef(np.argsort(np.argsort(lead)), np.argsort(np.argsort(energy)))[0, 1]:+.3f}")
    top = np.argsort(-np.abs(A[iu]))[:8]
    print("\nstrongest directed edges (i leads j; split-half values):")
    for t in top:
        i, j = iu[0][t], iu[1][t]
        print(f"  PC{i:2d} -> PC{j:2d}: {A[i, j]:+.4f}   h1 {A1[i, j]:+.4f}  h2 {A2[i, j]:+.4f}")
    print("\nnet leadingness by factor (sources +, sinks -):")
    for a in np.argsort(-lead):
        print(f"  PC{a:2d}: {lead[a]:+.4f}   signal {energy[a]:.1f}")
    pd.DataFrame({"pc": range(QPOOL), "net_lead": lead, "signal_energy": energy}
                 ).to_csv(f"{OUT}/factor_leadlag.csv", index=False)
    print(f"wrote {OUT}/factor_leadlag.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calib", "leadlag"], required=True)
    a = ap.parse_args()
    {"calib": stage_calib, "leadlag": stage_leadlag}[a.stage]()
