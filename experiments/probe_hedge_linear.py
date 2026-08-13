"""Hedge-over-penalty probe: exponential-weight aggregation across a penalty
grid, versus causal argmin selection, for the linear arms.

The tree bank established selection < best expert < hedge for experts. This
probe runs the same experiment for the linear penalty: at each retune
boundary (every 250 bars, embargo 25, tail 125 --- the identical schedule
to the tuners and to tree_hedge) candidates are weighted
w_k ~ exp(-eta * tail_loss_k), eta = sqrt(8 ln K / L) (Cesa-Bianchi &
Lugosi Thm 2.2, no free constant), and the forecast is the weighted average
of the per-penalty forecasts --- the average of linear forecasts, hence a
linear model whose coefficient vector is the weighted average of the
per-penalty ridge solutions. Selection/weighting uses a fresh fit-block
gram at each boundary (fit = window minus embargo minus tail), exactly the
tuners' forward-split convention; between boundaries the weights are fixed.

Arms (last 60k panel rows, per-bar refits, contract-identical scoring):
  a0_ols_har   reference benchmark
  ridge_tuned  argmin ridge over the battery grid (K=9)
  ridge_hedge  EWA over the same grid, full design
  blk2_fixed   two-block ridge at stated penalties (1, 100)
  blk2_tuned   argmin over the exog-block penalty grid (backbone a1=1)
  blk2_hedge   EWA over the same grid (backbone a1=1)
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

import numpy as np

import unification as U
import probe_multih as pm
from src.models.rolling_least_squares import RollingLeastSquares

WINDOW = 24000
TUNE_PER, EMBARGO, TAIL = 250, 25, 125
RIDGE_GRID = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
BLK2_GRID = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
A1 = 1.0


def _eta(K):
    return float(np.sqrt(8.0 * np.log(K) / TAIL))


class HedgeLinear:
    """K parallel exact rank-1 ridge solvers (one per penalty candidate);
    weights (or argmin selection) refreshed every TUNE_PER bars from
    fit-block-gram / forward-tail evaluation. HEDGE=False -> argmin."""

    HEDGE = True

    def __init__(self, F, y, pen_vecs, window):
        self.F, self.y = F, y
        self.pen_vecs = pen_vecs
        self.K = len(pen_vecs)
        self.solvers = []
        for _ in pen_vecs:
            s = RollingLeastSquares(alpha=0.0, fit_intercept=True)
            s.init_window(F[:window], y[:window])
            self.solvers.append(s)
        self.w = np.full(self.K, 1.0 / self.K)
        self.sel = self.K // 2
        self.trajectory = []

    def _beta(self, s, pv):
        gram = s._Sxx - np.outer(s._sx, s._sx) / s.n
        rhs = s._Sxy - s._sx * s._sy / s.n
        beta = np.linalg.solve(gram + np.diag(pv), rhs)
        return beta, s._sx / s.n, s._sy / s.n

    def predict(self, i, window, roll):
        if roll:
            x_in, y_in = self.F[i - 1], float(self.y[i - 1])
            x_out, y_out = self.F[i - 1 - window], float(self.y[i - 1 - window])
            for s in self.solvers:
                s.roll(x_in, y_in, x_out, y_out)
        preds = np.empty(self.K)
        for k, s in enumerate(self.solvers):
            beta, xbar, ybar = self._beta(s, self.pen_vecs[k])
            preds[k] = float(beta @ (self.F[i] - xbar) + ybar)
        return float(self.w @ preds) if self.HEDGE else float(preds[self.sel])

    def retune(self, i, window):
        """Fresh fit-block gram over rows [i-window, i-TAIL-EMBARGO), scored
        on the forward tail [i-TAIL, i) — the tuners' split, so candidate
        evaluation is never in-sample. One gram build, shared by all K."""
        flo, fhi = i - window, i - TAIL - EMBARGO
        vlo, vhi = i - TAIL, i
        Xf = self.F[flo:fhi]
        yf = self.y[flo:fhi]
        xbar = Xf.mean(0)
        ybar = float(yf.mean())
        Xc = Xf - xbar
        yc = yf - ybar
        G = Xc.T @ Xc
        c = Xc.T @ yc
        Xv = self.F[vlo:vhi] - xbar
        yv = self.y[vlo:vhi] - ybar
        losses = np.empty(self.K)
        for k, pv in enumerate(self.pen_vecs):
            beta = np.linalg.solve(G + np.diag(pv), c)
            losses[k] = float(np.sum((Xv @ beta - yv) ** 2))  # SUM over tail
        if self.HEDGE:
            eta = _eta(self.K)
            w = np.exp(-eta * (losses - losses.min()))
            self.w = w / w.sum()
            self.trajectory.append((i, self.w.copy(), losses.copy()))
        else:
            self.sel = int(np.argmin(losses))
            self.trajectory.append((i, self.sel, losses.copy()))


class ArgminRidge(HedgeLinear):
    HEDGE = False


def run_walk(model, n, window, lo, tag, t0):
    yh = np.full(n, np.nan)
    for i in range(lo, n):
        if i > lo and (i - lo) % TUNE_PER == 0:
            model.retune(i, window)
        yh[i] = model.predict(i, window, roll=(i > lo))
        if (i - lo) % 5000 == 0:
            print(f"  {tag} bar {i-lo}/{n-lo} {time.time()-t0:.0f}s", flush=True)
    return yh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, default=60000)
    ap.add_argument("--arm", default="all",
                    help="all (default) or one of ridge_hedge, ridge_tuned, blk2_hedge, blk2_tuned, blk2_fixed, a0_ols_har; single-arm mode writes its own npz")
    ap.add_argument("--out", default="results/hedge_lin")
    args = ap.parse_args()
    t0 = time.time()

    p = U._load_panel()
    n = len(p.y)
    lo = n - args.span
    rows = slice(lo - WINDOW, n)
    print(f"panel n={n}; span [{lo},{n})", flush=True)

    y = p.y[rows]
    y_raw = np.sqrt(p.rv_raw[rows] / p.baseline[rows])
    base = p.baseline[rows]
    ns = len(y)
    OFF = WINDOW

    spec_full = U.ARMS["br_tuned_all_features"]
    Ffull = np.ascontiguousarray(U._build_design(p, spec_full, WINDOW, arm="probe")[0][rows], dtype=np.float64)
    bb_idx = U._backbone_cols(p.names)
    ex_idx = U._exog_all_cols(p.names)
    H = np.ascontiguousarray(p.X[rows][:, bb_idx])
    E = np.ascontiguousarray(p.X[rows][:, ex_idx])
    spec0 = U.ARMS["a0_ols_har"]
    F0, kept0, _ = U._ols_design(p, spec0)
    F0 = F0[rows]
    p.X = None
    import gc; gc.collect()
    print(f"designs ready {time.time()-t0:.0f}s", flush=True)

    ARM = args.arm

    def _want(*names):
        return ARM == "all" or ARM in names

    losses = {}
    pv_full = [np.full(Ffull.shape[1], a) for a in RIDGE_GRID]
    if _want("ridge_hedge"):
        mh = HedgeLinear(Ffull, y, pv_full, WINDOW)
        losses["ridge_hedge"] = run_walk(mh, ns, WINDOW, OFF, "ridge_hedge", t0)
    if _want("ridge_tuned"):
        ma = ArgminRidge(Ffull, y, pv_full, WINDOW)
        losses["ridge_tuned"] = run_walk(ma, ns, WINDOW, OFF, "ridge_tuned", t0)
    del Ffull; gc.collect()

    nH = H.shape[1]
    if _want("blk2_hedge", "blk2_tuned", "blk2_fixed"):
        Fb = np.hstack([H, E]).astype(np.float64)
        pv_blk = []
        for a2 in BLK2_GRID:
            pv = np.empty(Fb.shape[1])
            pv[:nH] = A1
            pv[nH:] = a2
            pv_blk.append(pv)
        if _want("blk2_hedge"):
            mb_h = HedgeLinear(Fb, y, pv_blk, WINDOW)
            losses["blk2_hedge"] = run_walk(mb_h, ns, WINDOW, OFF, "blk2_hedge", t0)
        if _want("blk2_tuned"):
            mb_a = ArgminRidge(Fb, y, pv_blk, WINDOW)
            losses["blk2_tuned"] = run_walk(mb_a, ns, WINDOW, OFF, "blk2_tuned", t0)
        if _want("blk2_fixed"):
            pv_fixed = np.empty(Fb.shape[1])
            pv_fixed[:nH] = A1
            pv_fixed[nH:] = 100.0
            sf = RollingLeastSquares(alpha=0.0, fit_intercept=True)
            sf.init_window(Fb[:WINDOW], y[:WINDOW])
            yh = np.full(ns, np.nan)
            for i in range(OFF, ns):
                if i > OFF:
                    sf.roll(Fb[i - 1], y[i - 1], Fb[i - 1 - WINDOW], y[i - 1 - WINDOW])
                gram = sf._Sxx - np.outer(sf._sx, sf._sx) / sf.n
                beta = np.linalg.solve(gram + np.diag(pv_fixed),
                                       sf._Sxy - sf._sx * sf._sy / sf.n)
                yh[i] = float(beta @ (Fb[i] - sf._sx / sf.n) + sf._sy / sf.n)
                if (i - OFF) % 10000 == 0:
                    print(f"  blk2_fixed bar {i-OFF} {time.time()-t0:.0f}s", flush=True)
            losses["blk2_fixed"] = yh
        del Fb; gc.collect()
    del H, E; gc.collect()

    if _want("a0_ols_har"):
        yh0 = np.full(ns, np.nan)
        yh0[OFF:] = U._walk_ols(F0, y, WINDOW, OFF, ns, kept0)[0]
        losses["a0_ols_har"] = yh0
        del F0; gc.collect()
        print(f"a0 done {time.time()-t0:.0f}s", flush=True)

    import score_unification as su
    print("\n=== paired table (contract scoring, common scored bars) ===")
    L = {k: pm._contract_losses(v[OFF:], y_raw[OFF:], base[OFF:]) for k, v in losses.items()}
    b0 = L["a0_ols_har"]
    for tag in ["a0_ols_har", "ridge_tuned", "ridge_hedge", "blk2_fixed", "blk2_tuned", "blk2_hedge"]:
        l = L[tag]
        ok = np.isfinite(l) & np.isfinite(b0)
        if tag == "a0_ols_har":
            print(f"{tag:14s} QLIKE {np.nanmean(l[ok]):.5f}")
        else:
            dm = su.dm_test(l[ok], b0[ok], h=1)
            print(f"{tag:14s} QLIKE {np.nanmean(l[ok]):.5f}  dQ {np.nanmean(l[ok]-b0[ok]):+.5f}  DM {dm['dm']:+.2f}")
    for pair in [("ridge_hedge", "ridge_tuned"), ("blk2_hedge", "blk2_tuned"), ("blk2_hedge", "blk2_fixed"), ("ridge_hedge", "blk2_fixed")]:
        a, b = L[pair[0]], L[pair[1]]
        ok = np.isfinite(a) & np.isfinite(b)
        dm = su.dm_test(a[ok], b[ok], h=1)
        print(f"{pair[0]} vs {pair[1]}: dQ {np.mean(a[ok]-b[ok]):+.5f}  DM {dm['dm']:+.2f}")

    warr = np.array([t[1] for t in mh.trajectory])
    wblk = np.array([t[1] for t in mb_h.trajectory])
    print("\nridge hedge: mean weights by grid point:")
    for g, wv in zip(RIDGE_GRID, warr.mean(0)):
        print(f"  alpha={g:>7g}: {wv:.3f}")
    print("blk2 hedge: mean weights by grid point:")
    for g, wv in zip(BLK2_GRID, wblk.mean(0)):
        print(f"  alpha={g:>7g}: {wv:.3f}")
    print("ridge hedge top-grid weight over retunes (first 10):", np.round(warr[:10, -1], 3))
    sel_arr = [t[1] for t in ma.trajectory]
    print("argmin selections (first 14):", sel_arr[:14])

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "hedge_lin.npz" if args.arm == "all" else f"hedge_lin_{args.arm}.npz"),
             **{f"loss_{k}": v for k, v in L.items()},
             ridge_w=warr, blk2_w=wblk)
    print(f"wrote {args.out}/hedge_lin.npz")


if __name__ == "__main__":
    main()
