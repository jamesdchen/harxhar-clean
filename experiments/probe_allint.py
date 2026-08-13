"""All-interactions probe: drop the top-100 curation, keep ALL pairwise
products, estimate the interaction block with the recursive elastic net
(reclasso_har.enet_online, exact warm rank-1 homotopy per bar).

Question: the curated product block keeps 100 of 8,911 candidate products by
a one-shot correlation screen. Does keeping all 8,911 under a causal
selection estimator beat it? Selection is what makes the wide block cheap:
the active set stays small, so per-bar cost is O(m.|A|), not O(p^3).

Design (three groups, same information set as blk3_user):
  G1 backbone (~52): locked --- unpenalized, never leaves the active set
                       (intercept + HAR convention of enet_online).
  G2 exog (~529):    ridge --- mu=0, lam2=100 on the gram diagonal
                       (campaign ridge units, matches blk3_user's alpha=100).
  G3 interactions (8,911): elastic net --- mu = n.a.rho, lam2 = n.a.(1-rho)
                       (sklearn ElasticNet units, reclasso_har mapping),
                       same causal floored rolling-SD column scaling as the
                       curated block.

Arms: a0_ols_har (reference), blk3_user (curated-100 ridge, reference),
allint_fixed ((a,rho) selected ONCE on the initial window's forward split,
then held --- the campaign's fixed-penalty control style, disclosed);
--tuned adds allint_tuned (rho=1.0 mu-path retune every TUNE_PER bars;
alphas read off one homotopy path per retune).

Scoring: the identical protocol as probe_multih (MZ + EWMA second moment).
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
from src.models.reclasso_har import (
    enet_coef,
    enet_online,
    forward_window_split,
    lasso_path_coefs,
)

WINDOW = 24000
SCALE_MIN_PERIODS = 1000
TUNE_PER = 2500          # tuned arm cadence; fixed arm re-anchors every 250 (exact active-set re-solve + KKT scan) (disclosed: 10x sparser than the 250 convention)
VAL_TAIL, EMBARGO = 125, 25
ALPHA_GRID = tuple(float(a) for a in np.logspace(-5, -3, 5))
EXOG_LAM2 = 100.0        # exog block ridge, campaign gram units
CD_CYCLES = 4


# ── interaction design ────────────────────────────────────────────────────

def _build_products(B):
    """All triu products (incl. squares) of the base cols. f32."""
    n, p = B.shape
    m = p * (p + 1) // 2
    P = np.empty((n, m), dtype=np.float32)
    pos = 0
    for j in range(p):
        k = p - j
        P[:, pos:pos + k] = (B[:, j:j + 1] * B[:, j:]).astype(np.float32)
        pos += k
    return P


def _build_products_scaled(B, window, col_chunk=1000):
    """All triu products of the base cols with the causal floored rolling-SD
    scale applied, STREAMING by column chunk: peak memory is one chunk's
    cumsums, not the full (n x 8911) product and scale matrices."""
    n, p = B.shape
    m = p * (p + 1) // 2
    PS = np.empty((n, m), dtype=np.float32)
    SCALE_MIN = 1000
    for c0 in range(0, m, col_chunk):
        # locate the base-pair range for this output chunk: columns are
        # j-major (j, j..p-1); find the j-span covering [c0, c1)
        c1 = min(c0 + col_chunk, m)
        # j starts: cumsum of (p - j) >= c0
        starts = np.concatenate([[0], np.cumsum([p - j for j in range(p)])])
        j0 = int(np.searchsorted(starts, c0, side="right") - 1)
        j1 = int(np.searchsorted(starts, c1 - 1, side="right") - 1)
        lo = int(starts[j0])
        hi = int(starts[j1 + 1])
        Pc = np.empty((n, hi - lo), dtype=np.float32)
        pos = 0
        for j in range(j0, j1 + 1):
            k = p - j
            Pc[:, pos:pos + k] = (B[:, j:j + 1] * B[:, j:]).astype(np.float32)
            pos += k
        Pf = np.nan_to_num(Pc, nan=0.0).astype(np.float64)
        cs1 = np.vstack([np.zeros((1, hi - lo)), np.cumsum(Pf, axis=0)])
        cs2 = np.vstack([np.zeros((1, hi - lo)), np.cumsum(Pf * Pf, axis=0)])
        s1 = cs1[window:] - cs1[:-window]
        s2 = cs2[window:] - cs2[:-window]
        mean = s1 / window
        var = np.maximum(s2 / window - mean * mean, 0.0)
        sd = np.sqrt(var).astype(np.float32)
        S = np.full((n, hi - lo), np.nan, dtype=np.float32)
        S[window:n] = sd[: n - window]     # shift(1): row t uses trailing [t-window, t-1]
        S[:window] = S[window]
        med = np.nanmedian(S, axis=1, keepdims=True)
        S = np.maximum(S, 0.1 * med)
        bad = ~np.isfinite(S) | (S <= 0)
        S[bad] = 1.0
        PS[:, lo:hi] = (Pc / S).astype(np.float32)
        del Pc, Pf, cs1, cs2, sd, S
    return PS

# ── the all-interactions enet walker ──────────────────────────────────────

class AllIntEnet:
    """Warm per-bar elastic-net walk over [1 | H locked | E ridge | P enet].

    mu_vec = [0 | 0 | 0 | n.a.rho]; gram-diag lam2 = [0 | 0 | EXOG_LAM2 |
    n.a.(1-rho)]. Per-bar: two enet_online calls (add newest row, drop
    oldest). Re-anchor at every retune: exact gram rebuild from the current
    window (kills rank-1 drift, the GramState lesson).
    """

    def __init__(self, nH, nE, nP):
        self.nH, self.nE, self.nP = nH, nE, nP
        self.m = nH + nE + nP + 1
        self.locked = np.zeros(self.m, dtype=bool)
        self.locked[: nH + 1] = True  # intercept col 0 + backbone
        self.alpha_, self.l1_ = ALPHA_GRID[len(ALPHA_GRID) // 2], 1.0
        self.trace = []

    def _design(self, H, E, PS):
        return np.hstack([np.ones((H.shape[0], 1)), H, E, PS]).astype(np.float64)

    def _pen(self, n):
        mu_vec = np.zeros(self.m)
        lam2_vec = np.zeros(self.m)
        i0 = 1 + self.nH + self.nE
        lam2_vec[1 + self.nH:i0] = EXOG_LAM2
        mu_vec[i0:] = n * self.alpha_ * self.l1_
        lam2_vec[i0:] = n * self.alpha_ * (1.0 - self.l1_)
        return mu_vec, lam2_vec

    def _fwl(self, X, y):
        """Residualize y, E, P on the locked block (intercept+backbone)."""
        iH1 = 1 + self.nH
        iE1 = iH1 + self.nE
        Hh, Ee, Pp = X[:, :iH1], X[:, iH1:iE1], X[:, iE1:]
        Hp = np.linalg.pinv(Hh)
        bHy, bHE, bHP = Hp @ y, Hp @ Ee, Hp @ Pp
        return (Ee - Hh @ bHE, Pp - Hh @ bHP, y - Hh @ bHy, bHy, bHE, bHP)

    def _batch_fit(self, X, y, alpha, l1, cached=None):
        """Block CD: exog ridge <-> interaction enet, CD_CYCLES cycles.
        cached = (Eres, Pres, yres, bHy, bHE, bHP, GE, GP) to skip rebuilds."""
        n = len(y)
        if cached is None:
            Eres, Pres, yres, bHy, bHE, bHP = self._fwl(X, y)
            GE = Eres.T @ Eres
            GE[np.diag_indices_from(GE)] += EXOG_LAM2
            GP = Pres.T @ Pres
            cached = (Eres, Pres, yres, bHy, bHE, bHP, GE, GP)
        Eres, Pres, yres, bHy, bHE, bHP, GE, GP = cached
        bE = np.zeros(self.nE)
        bP = np.zeros(self.nP)
        for _ in range(CD_CYCLES):
            rE = yres - Pres @ bP
            bE = np.linalg.solve(GE, Eres.T @ rE)
            rP = yres - Eres @ bE
            bP = enet_coef(GP, Pres.T @ rP, n, alpha, l1)
        th = np.zeros(self.m)
        th[1 + self.nH:1 + self.nH + self.nE] = bE
        th[1 + self.nH + self.nE:] = bP
        th[:1 + self.nH] = bHy - bHE @ bE - bHP @ bP
        return th

    def seed(self, H, E, PS, y):
        X = self._design(H, E, PS)
        th = self._batch_fit(X, y, self.alpha_, self.l1_)
        n = len(y)
        self.mu_vec, lam2_vec = self._pen(n)
        self.Gr = X.T @ X
        self.Gr[np.diag_indices_from(self.Gr)] += lam2_vec
        self.c = X.T @ y
        self.th = th
        self.A = list(np.where((np.abs(th) > 1e-9) | self.locked)[0])
        self.s = [0.0 if self.locked[j] else float(np.sign(th[j])) for j in self.A]
        return self

    def tune(self, H, E, PS, y):
        """Forward split; rho=1.0 mu-path (all alphas off one path); plus a
        rho=0.5 bracket at the path-selected alpha. Select by tail MSE of the
        FULL reassembled model."""
        n = len(y)
        fl, fh, vl, vh = forward_window_split(n, n, VAL_TAIL, EMBARGO)
        Xf = self._design(H[fl:fh], E[fl:fh], PS[fl:fh])
        yf = y[fl:fh]
        Xv = self._design(H[vl:vh], E[vl:vh], PS[vl:vh])
        yv = y[vl:vh]
        cached = self._fwl(Xf, yf)
        Eres, Pres, yres, bHy, bHE, bHP = cached
        GE = Eres.T @ Eres
        GE[np.diag_indices_from(GE)] += EXOG_LAM2
        GP = Pres.T @ Pres
        cached_fit = (Eres, Pres, yres, bHy, bHE, bHP, GE, GP)

        cP = Pres.T @ yres  # lasso path starts from CD cycle 0 (bE=0)
        mu_grid = np.array([n * a * 1.0 for a in sorted(ALPHA_GRID, reverse=True)])
        path = lasso_path_coefs(GP, cP, mu_grid)  # (n_alpha, nP)

        def assemble(bP, alpha, rho):
            bE = np.zeros(self.nE)
            for _ in range(CD_CYCLES):
                rE = yres - Pres @ bP
                bE = np.linalg.solve(GE, Eres.T @ rE)
                if rho == 1.0:
                    break  # path bP already at rho=1; one exog update suffices
                rP = yres - Eres @ bE
                bP = enet_coef(GP, Pres.T @ rP, n, alpha, rho)
            th = np.zeros(self.m)
            th[1 + self.nH:1 + self.nH + self.nE] = bE
            th[1 + self.nH + self.nE:] = bP
            th[:1 + self.nH] = bHy - bHE @ bE - bHP @ bP
            return th

        cands = []
        for k, a in enumerate(sorted(ALPHA_GRID, reverse=True)):
            cands.append((a, 1.0, path[k]))
        best = None
        for a, rho, bP in cands:
            th = assemble(bP, a, rho)
            mse = float(np.mean((Xv @ th - yv) ** 2))
            if best is None or mse < best[0]:
                best = (mse, a, rho)
        # rho=0.5 bracket at the selected alpha (one batch fit, disclosed)
        _, a1, _ = best
        th05 = self._batch_fit(Xf, yf, a1, 0.5, cached=cached_fit)
        mse05 = float(np.mean((Xv @ th05 - yv) ** 2))
        if mse05 < best[0]:
            best = (mse05, a1, 0.5)
        _, a, rho = best
        self.alpha_, self.l1_ = a, rho
        self.trace.append((a, rho))
        return a, rho

    def reanchor(self):
        """Exact active-set re-solve + full KKT scan (cheap: one |A|^3 solve
        + one Gr matvec). Any inactive KKT violation forces a loud full
        re-seed by the caller --- never silently drifts. Returns n_violations."""
        Aa = np.asarray(self.A, dtype=int)
        sA = np.asarray(self.s, dtype=float)
        M = self.Gr[np.ix_(Aa, Aa)]
        rhs = self.c[Aa] - self.mu_vec[Aa] * sA
        th = np.zeros(self.m)
        th[Aa] = np.linalg.solve(M, rhs)
        th[~np.isfinite(th)] = 0.0
        r = self.c - self.Gr @ th
        viol = 0
        for j in range(self.m):
            if self.locked[j] or j in set(self.A):
                continue
            if abs(r[j]) > self.mu_vec[j] + 1e-7:
                viol += 1
        self.th = th
        return viol

    def roll(self, x_in, y_in, x_out, y_out):
        self.th, self.A, self.s = enet_online(
            self.Gr, self.c, self.mu_vec, x_in, float(y_in), +1.0,
            self.th, self.A, self.s, self.locked)
        self.th, self.A, self.s = enet_online(
            self.Gr, self.c, self.mu_vec, x_out, float(y_out), -1.0,
            self.th, self.A, self.s, self.locked)


# ── main walk ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, default=60000)
    ap.add_argument("--nslices", type=int, default=1)
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--out", default="results/allint")
    args = ap.parse_args()

    t0 = time.time()
    p = U._load_panel()
    n = len(p.y)
    per = args.span // args.nslices
    lo = n - args.span + args.slice * per
    hi = lo + per if args.slice < args.nslices - 1 else n
    print(f"panel n={n}; slice {args.slice}/{args.nslices}: eval span [{lo},{hi})", flush=True)

    bb_idx = U._backbone_cols(p.names)
    ex_idx = U._exog_all_cols(p.names)
    base_idx = U._product_base_cols(p.names)
    print(f"blocks: backbone {len(bb_idx)}, exog {len(ex_idx)}, base {len(base_idx)}", flush=True)

    lo2 = max(0, lo - 2 * WINDOW)   # 2W history: the product columns' rolling
    OFF = lo - lo2                  # scales need one window of trailing rows
    rows = slice(lo2, hi)
    H = np.ascontiguousarray(p.X[rows][:, bb_idx])
    E = np.ascontiguousarray(p.X[rows][:, ex_idx])
    B = np.ascontiguousarray(p.X[rows][:, base_idx], dtype=np.float32)
    y = p.y[rows]
    y_raw = np.sqrt(p.rv_raw[rows] / p.baseline[rows])
    base_diag = p.baseline[rows]
    nH, nE = H.shape[1], E.shape[1]
    ns = H.shape[0]
    losses = {}

    def score(yhat):
        return pm._contract_losses(yhat, y_raw, base_diag)

    # Build reference designs and slice them BEFORE the products: the panel
    # X (2.6 GB) is freed before the streaming product build (8G slot).
    spec0 = U.ARMS["a0_ols_har"]
    F0, kept0, _ = U._ols_design(p, spec0)
    F0 = F0[rows]
    spec3 = U.ARMS["blk3_user"]
    F3, a3 = U._build_design(p, spec3, WINDOW, arm="blk3_user")
    F3 = F3[rows]
    p.X = None
    import gc; gc.collect()
    print(f"reference designs sliced; panel freed at {time.time()-t0:.0f}s", flush=True)

    print(f"building all interactions for {ns} rows...", flush=True)
    PS = _build_products_scaled(B, WINDOW)
    del B
    gc.collect()
    print(f"interaction design {PS.shape} built in {time.time()-t0:.0f}s", flush=True)
    nP = PS.shape[1]

    yh0 = np.full(ns, np.nan)
    yh0[OFF:] = U._walk_ols(F0, y, WINDOW, OFF, ns, kept0)[0]
    losses["a0_ols_har"] = score(yh0[OFF:])
    del F0
    print(f"a0 done {time.time()-t0:.0f}s", flush=True)

    yh3 = np.full(ns, np.nan)
    yh3[OFF:] = U._walk_ridge(F3, y, WINDOW, OFF, ns, alpha=a3)
    losses["blk3_user"] = score(yh3[OFF:])
    del F3
    print(f"blk3_user done {time.time()-t0:.0f}s", flush=True)

    def run_allint(tuned):
        w = AllIntEnet(nH, nE, nP)
        a, rho = w.tune(H[OFF-WINDOW:OFF], E[OFF-WINDOW:OFF], PS[OFF-WINDOW:OFF], y[OFF-WINDOW:OFF])
        print(f"  initial tune: alpha={a:.1e} rho={rho}", flush=True)
        w.seed(H[OFF-WINDOW:OFF], E[OFF-WINDOW:OFF], PS[OFF-WINDOW:OFF], y[OFF-WINDOW:OFF])
        yh = np.full(ns, np.nan)
        for i in range(OFF, ns):
            if i > OFF and (i - OFF) % TUNE_PER == 0:
                if tuned:
                    a, rho = w.tune(H[i-WINDOW:i], E[i-WINDOW:i], PS[i-WINDOW:i], y[i-WINDOW:i])
                    w.seed(H[i-WINDOW:i], E[i-WINDOW:i], PS[i-WINDOW:i], y[i-WINDOW:i])
                elif (i - OFF) % 250 == 0:
                    v = w.reanchor()
                    if v:
                        print(f"  KKT violations at bar {i-OFF}: {v} -> full re-seed", flush=True)
                        w.seed(H[i-WINDOW:i], E[i-WINDOW:i], PS[i-WINDOW:i], y[i-WINDOW:i])
            if i > OFF:
                xi = np.concatenate(([1.0], H[i-1], E[i-1], PS[i-1])).astype(np.float64)
                xo = np.concatenate(([1.0], H[i-WINDOW-1], E[i-WINDOW-1], PS[i-WINDOW-1])).astype(np.float64)
                w.roll(xi, y[i-1], xo, y[i-WINDOW-1])
            xp = np.concatenate(([1.0], H[i], E[i], PS[i])).astype(np.float64)
            yh[i] = float(w.th @ xp)
            if (i - OFF) % 5000 == 0:
                print(f"  bar {i-WINDOW}/{ns-WINDOW} |A|={len(w.A)} {time.time()-t0:.0f}s", flush=True)
                np.savez(os.path.join(args.out, f"ckpt_{'t' if tuned else 'f'}.npz"), yh_partial=yh)
        return yh[OFF:], w

    yh_f, wf = run_allint(tuned=False)
    losses["allint_fixed"] = score(yh_f)
    print(f"allint_fixed done {time.time()-t0:.0f}s (init a={wf.alpha_:.1e} rho={wf.l1_})", flush=True)
    if args.tuned:
        yh_t, wt = run_allint(tuned=True)
        losses["allint_tuned"] = score(yh_t)
        print(f"allint_tuned done {time.time()-t0:.0f}s; trace={wt.trace[:8]}", flush=True)

    import score_unification as su
    print("\n=== paired table (common scored bars) ===")
    base = losses["a0_ols_har"]
    for tag, L in losses.items():
        ok = np.isfinite(L) & np.isfinite(base)
        if tag == "a0_ols_har":
            print(f"{tag:14s} QLIKE {np.nanmean(L[ok]):.5f}")
        else:
            dm = su.dm_test(L[ok], base[ok], h=1)
            print(f"{tag:14s} QLIKE {np.nanmean(L[ok]):.5f}  dQ {np.nanmean(L[ok]-base[ok]):+.5f}  DM {dm['dm']:+.2f}")
    for tag in ("allint_fixed", "allint_tuned"):
        if tag in losses:
            L, C = losses[tag], losses["blk3_user"]
            ok = np.isfinite(L) & np.isfinite(C)
            dm = su.dm_test(L[ok], C[ok], h=1)
            print(f"{tag} vs blk3_user: dQ {np.mean(L[ok]-C[ok]):+.5f}  DM {dm['dm']:+.2f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"allint_s{args.slice:02d}.npz" if args.nslices > 1 else os.path.join(args.out, "allint.npz"))
    np.savez(path, **{f"loss_{k}": v for k, v in losses.items()})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
