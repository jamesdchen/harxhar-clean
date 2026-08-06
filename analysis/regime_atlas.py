"""§34: the regime atlas — the professor's regime-extraction experiment replaced by its
intended deliverable. A1 coefficient-trajectory segmentation; A2 decomposition of
spectral-cluster labels onto measured coordinates; A3 labels -> amplitude uplift (gated);
A4 is assembled in the writeup from the record."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_law import _blocks  # noqa: E402
from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.cucuringu import _daily, _antisym, _rot_plane  # noqa: E402
from analysis.map_monitor import _frame_and_scores  # noqa: E402
from analysis.synthesis import _p  # noqa: E402

OUT = "results/alpha_manifestation"


def stage_a1() -> None:
    """Monthly coefficient snapshots of the one-stage design; C4 scan on the exog block."""
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    day_codes = pd.factorize(ts.dt.normalize())[0]
    XH, XL, XS, P = _blocks(p)
    A = 3000.0
    X = np.hstack([XH * np.sqrt(A), XL, XS, P * np.sqrt(0.1)])
    n, pc = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    starts = np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1]])
    bounds = np.r_[starts, n]
    ndays = len(starts)
    reg = np.diag(np.r_[np.full(pc, A), 0.0])
    G = np.zeros((pc + 1, pc + 1))
    bv = np.zeros(pc + 1)
    added = lo = 0
    betas, when = [], []
    y = np.nan_to_num(p.y)
    lo_ex, hi_ex = XH.shape[1], XH.shape[1] + XL.shape[1]  # exog block within beta
    for d in range(251, ndays):
        hi = d - 1
        while added < hi:
            s, t = bounds[added], bounds[added + 1]
            Z = Xa[s:t]
            G += Z.T @ Z
            bv += Z.T @ y[s:t]
            added += 1
        while added - lo > 250:
            s, t = bounds[lo], bounds[lo + 1]
            Z = Xa[s:t]
            G -= Z.T @ Z
            bv -= Z.T @ y[s:t]
            lo += 1
        if (d - 251) % 21 == 0:
            betas.append(np.linalg.solve(G + reg, bv)[lo_ex:hi_ex])
            when.append(str(ts.iloc[bounds[d]].date()))
    B = np.array(betas)
    print(f"A1: {len(B)} monthly exog-coefficient snapshots ({B.shape[1]} coefs)")
    S = np.corrcoef(B)
    w = 12
    scan = np.array([S[t - w : t, t : t + w].mean() for t in range(w, len(B) - w)])
    thr = scan.mean() - 1.5 * scan.std()
    print(f"    scan mean {scan.mean():+.3f} sd {scan.std():.3f}; breaks (<mean-1.5sd, local min):")
    for t in range(1, len(scan) - 1):
        if scan[t] < thr and scan[t] <= scan[t - 1] and scan[t] <= scan[t + 1]:
            print(f"      {when[t + w]}  sim {scan[t]:+.3f}")
    h = len(B) // 2
    def seg(b):
        s = np.corrcoef(b)
        return np.array([s[t - w : t, t : t + w].mean() for t in range(w, len(b) - w)])
    pd.DataFrame({"asof": when, "beta_norm": np.linalg.norm(B, axis=1)}).to_csv(
        f"{OUT}/regime_atlas_a1.csv", index=False)
    print(f"    wrote {OUT}/regime_atlas_a1.csv")


def stage_a23() -> None:
    """A2: decompose spectral-cluster labels onto measured coordinates. A3: labels->amplitude."""
    rng = np.random.default_rng(5)
    G, e, ts = _frame_and_scores()
    Gd, ed, tsd = _daily(G, e, ts)
    nd = len(Gd)
    # measured coordinates: log amplitude, daily-cycle phase, secular era
    a = pd.Series(e**2).groupby(ts.dt.normalize().values).mean().to_numpy()
    la = np.log(a + 1e-12)
    A = _antisym(Gd)
    u, w = _rot_plane(A)
    phi = np.arctan2(Gd @ w, Gd @ u)
    era = (pd.to_datetime(tsd.dt.normalize()).astype("int64")
           // 86_400_000_000_000).to_numpy().astype(float)
    print("A2: spectral-cluster labels decomposed onto {log-amp, phase, era}:")
    Xdec = np.column_stack([la, np.cos(phi), np.sin(phi), era / era.max()])
    Xdec = (Xdec - Xdec.mean(0)) / (Xdec.std(0) + 1e-12)
    for k in (2, 3, 5, 8):
        lab = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=25,
                                 random_state=0, assign_labels="discretize").fit(Gd).labels_
        onehot = np.eye(k)[lab]
        r2s = []
        for j in range(k):
            yj = onehot[:, j] - onehot[:, j].mean()
            b = np.linalg.lstsq(np.c_[np.ones(nd), Xdec], yj, rcond=None)[0]
            pred = np.c_[np.ones(nd), Xdec] @ b
            r2s.append(1 - ((yj - pred) ** 2).sum() / (yj**2).sum())
        share = float(np.mean(r2s))
        print(f"    k={k}: mean label R2 from measured coords {share:.3f}")

        if k == 5:
            print("A3: labels -> amplitude uplift (gate: uplift > 0 both halves, > null p95):")
            X0 = pd.DataFrame({"l1": pd.Series(la).shift(1),
                               "m5": pd.Series(la).shift(1).rolling(5).mean(),
                               "m21": pd.Series(la).shift(1).rolling(21).mean()}).to_numpy()
            lab1 = np.roll(lab, 1).astype(float)  # yesterday's regime label, causal
            OH = np.eye(k)[lab1.astype(int)][:, :-1]
            for tname, tgt in (("next-day", la),
                               ("next-21d", pd.Series(la).rolling(21).mean().shift(-20).to_numpy())):
                m = np.isfinite(X0).all(1) & np.isfinite(tgt)
                idx = np.flatnonzero(m)
                h = len(idx) // 2
                ups, nulls = [], []
                for tr, te in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
                    def r2(Xtr, Xte):
                        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Xtr[tr]], tgt[tr],
                                            rcond=None)[0]
                        pr = np.c_[np.ones(len(te)), Xte[te]] @ b
                        return 1 - ((tgt[te] - pr) ** 2).sum() / ((tgt[te] - tgt[te].mean()) ** 2).sum()
                    base = r2(X0, X0)
                    full = r2(np.c_[X0, OH], np.c_[X0, OH])
                    ups.append(full - base)
                    nv = []
                    for _ in range(200):
                        OHs = np.roll(OH, int(rng.integers(63, nd - 63)), axis=0)
                        nv.append(r2(np.c_[X0, OHs], np.c_[X0, OHs]) - base)
                    nulls.append(float(np.quantile(nv, 0.95)))
                ok = all(u > 0 for u in ups) and all(u > q for u, q in zip(ups, nulls))
                print(f"    {tname:9s}: uplift {ups[0]:+.4f}/{ups[1]:+.4f} "
                      f"(null p95 {nulls[0]:+.4f}/{nulls[1]:+.4f})  "
                      f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a1", "a23"], required=True)
    a = ap.parse_args()
    {"a1": stage_a1, "a23": stage_a23}[a.stage]()
