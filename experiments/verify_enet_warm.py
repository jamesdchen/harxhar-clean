"""Verify enet_warm (warm active-set online update) == enet_coef (verified homotopy), and time the
streaming warm-chain. Uses the real Schur-FWL exog Gram (HAR unpenalized) on all_buckets windows."""

import json
import time

import numpy as np

from src.models.reclasso_har import GramState, enet_coef, enet_warm

B = "results/covid_imp_rank/all_buckets"
HAR = {"har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"}
feats = json.load(open(f"{B}/meta.json"))["feats"]
hm = np.array([f in HAR for f in feats])
em = ~hm
iH, iE = np.where(hm)[0], np.where(em)[0]
X = np.load(f"{B}/X_imp.npy")
y = np.load(f"{B}/y.npy")
GRID = [(1e-3, 0.2), (1e-3, 0.5), (3e-3, 0.4), (1e-2, 0.8), (1e-3, 1.0)]


def schur_exog(G, c, n):
    GHH = G[np.ix_(iH, iH)]
    GHE = G[np.ix_(iH, iE)]
    GEE = G[np.ix_(iE, iE)]
    A = np.linalg.solve(GHH, GHE)
    b = np.linalg.solve(GHH, c[iH])
    return GEE - GHE.T @ A, c[iE] - GHE.T @ b


def centered_gram(Xw, yw):
    xb, yb = Xw.mean(0), float(yw.mean())
    Xc, yc = Xw - xb, yw - yb
    return Xc.T @ Xc, Xc.T @ yc


# ---- Test 1: cold enet_warm == enet_coef ----
tw = 48000
t_r = 175000
Xw = X[t_r - tw : t_r].astype(np.float64)
yw = y[t_r - tw : t_r].astype(np.float64)
G, c = centered_gram(Xw, yw)
Ger, cer = schur_exog(G, c, tw)
print("=== Test 1: cold warm-solve vs homotopy ===")
worst = 0.0
for a, l1 in GRID:
    ref = enet_coef(Ger, cer, tw, a, l1)
    th, A, s = enet_warm(Ger, cer, tw, a, l1)  # cold (no warm start)
    d = float(np.max(np.abs(ref - th)))
    worst = max(worst, d)
    print(f"  a={a:g} l1={l1:g}: max|diff|={d:.2e}  nnz={len(A)}")
print(f"  WORST cold diff = {worst:.2e}  {'PASS' if worst < 1e-6 else 'FAIL'}")

# ---- Test 2: warm from a DIFFERENT config's active set ----
print("\n=== Test 2: warm-started from a neighbor's (A,s) ==")
_, A0, s0 = enet_warm(Ger, cer, tw, 1e-3, 0.5)
worst = 0.0
for a, l1 in GRID:
    ref = enet_coef(Ger, cer, tw, a, l1)
    th, A, s = enet_warm(Ger, cer, tw, a, l1, active=A0, signs=s0)
    d = float(np.max(np.abs(ref - th)))
    worst = max(worst, d)
    print(f"  a={a:g} l1={l1:g}: max|diff|={d:.2e}")
print(f"  WORST warm diff = {worst:.2e}  {'PASS' if worst < 1e-6 else 'FAIL'}")

# ---- Test 3: streaming warm-chain vs fresh per bar, + timing ----
print("\n=== Test 3: streaming (slide window 1 bar), warm-chain vs fresh homotopy ===")
a, l1 = 1e-3, 0.2  # the dense incumbent (hardest)
nbars = 400
start = t_r
gs = GramState.from_block(Xw, yw)
A, s = None, None
maxdiff = 0.0
t_warm = t_fresh = 0.0
for k in range(nbars):
    t = start + k
    # slide: add row t, remove the oldest (t-tw)
    if k > 0:
        gs.slide(X[t - 1].astype(np.float64), float(y[t - 1]), X[t - 1 - tw].astype(np.float64), float(y[t - 1 - tw]))
    inv_n = 1.0 / gs.n
    Gc = gs.g - np.outer(gs.sx, gs.sx) * inv_n
    cc = gs.c - gs.sx * (gs.sy * inv_n)
    Ger, cer = schur_exog(Gc, cc, gs.n)
    t0 = time.time()
    th, A, s = enet_warm(Ger, cer, gs.n, a, l1, active=A, signs=s)
    t_warm += time.time() - t0
    t0 = time.time()
    ref = enet_coef(Ger, cer, gs.n, a, l1)
    t_fresh += time.time() - t0
    maxdiff = max(maxdiff, float(np.max(np.abs(th - ref))))
print(f"  bars={nbars}  max|warm - fresh| = {maxdiff:.2e}  {'PASS' if maxdiff < 1e-6 else 'FAIL'}")
print(f"  time: warm {t_warm:.1f}s ({1000*t_warm/nbars:.1f} ms/bar)  fresh {t_fresh:.1f}s ({1000*t_fresh/nbars:.1f} ms/bar)")
print(f"  speedup {t_fresh/t_warm:.1f}x  ->  every-bar all_buckets ~{74934*t_warm/nbars/60:.0f} min/config (warm)")
