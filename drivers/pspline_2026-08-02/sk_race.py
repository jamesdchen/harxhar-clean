# ruff: noqa: E402, E741, E731  (experiment driver, verified by execution)
"""Self-kernel basis race: smooth-beta (P-spline) vs ladder vs parametric kernels.

The smooth-coefficient-function frame predicts the target self-kernel
beta(log u) is smooth, HAR ladders are step-function quadratures of it, and a
roughness-penalized spline at matched effective df should TIE the crowned b2
ladder (receipt calibrating the prior: a single fitted power law ~ reproduces
the 12-rung ladder). This driver races, on identical support (lags 1..2048,
b5 to 3125), identical calendar block, identical harness (W=24000 sliding
ridge, cadence-240 segments, per-bar rank-1 gram updates, Duan smearing ->
per-bar QLIKE -> DM vs the b2 arm):

  b5      production ladder [1,5,25,125,625,3125], ridge alpha=1 (anchor:
          must reproduce the known b2-beats-b5 gap, ~ -0.0008 at scale)
  b2      crowned ladder [1,2,...,2048] (12 rungs) — THE comparator
  b2prep  same, but from the prep cache's rolling-robust-scaled har_ma_*
          columns (validates the in-driver feature rebuild)
  arith12 12 rungs EQUALLY spaced in lag level (1..2048) — same df as b2,
          spacing test: the smooth-in-LOG-lag prior says most curvature
          lives at short lags, so level spacing should lose clearly
  bsq2    base-sqrt(2) ladder, 22 rungs on the same [1,2048] support —
          the diminishing-returns-past-b2 check (does refinement asymptote?)
  spl     cubic B-spline basis (16 fns) on log2-lag in [0,11], second-
          difference penalty on the coefficients; HEADLINE lambda is
          matched to the b2 arm's effective df on the first window
          (pre-registered comparison); SPL_SWEEP=1 adds a lambda curve
  pl      single power-law kernel u^-beta, beta refit per segment by SSE
          (prof_series convention), at U=256 and U=2048
  exp     single exponential kernel exp(-u/tau), tau refit per segment
  mix5    fixed 5-beta power-law mixture (0.3,0.6,1.0,1.5,2.2), U=256,
          ridge-weighted (the Hawkes-native compressed form, for context)

Penalty null-space note: D2 on the spline coefs leaves functions LINEAR in
log2-lag (level scale) unpenalized — that is not the power law (which is
linear in LOG-level); stated here so nobody reads lambda->inf as "recovers
the power-law arm".

env: PREP (cache npz), TILES ("1,2"), ARMS (comma list; default all),
     SPL_SWEEP, CADENCE (240)
Outputs: results/pspline_race/{preds_*.npz, summary printed}. The b2
comparator always runs (it is the DM reference); post-hoc DMs between any
two arms can be recomputed from the saved per-bar preds.
"""

import os
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics

PREP = os.environ.get("PREP", "results/prep_cache_all_features_b2_r32000.npz")
CADENCE = int(os.environ.get("CADENCE", "240"))
TILES = [int(v) for v in os.environ.get("TILES", "1,2").split(",")]
W = 24000
BURN = 3125          # raw rows dropped by the prep burn-in
OUT = "results/pspline_race"
os.makedirs(OUT, exist_ok=True)

z = np.load(PREP, allow_pickle=True)
X, y, b = z["X"], z["y"], z["baselines"]
names = [str(s) for s in z["names"]]
n = len(y)
CAL = X[:, [names.index(c) for c in
            ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4",
             "hour", "is_overnight", "is_open", "is_close"]]]

# ── pre-drop adjusted target (full history for the first window's lags) ──
ADJ_CACHE = f"{OUT}/adj_full.npy"
if os.path.exists(ADJ_CACHE):
    s_full = np.load(ADJ_CACHE)
else:
    from src.backtest.executor import load_and_transform
    from src.data.loading import get_bucket

    df, _ = load_and_transform(
        "data", get_bucket("baseline"), target_use_diurnal=True,
        target_winsor_window=240, dropna_with_exog=False,
        overnight_fill=True, impute_indicate=True,
    )
    s_full = df["adj_RV"].values.astype(np.float64)[: BURN + n]
    np.save(ADJ_CACHE, s_full)
err = np.nanmax(np.abs(s_full[BURN : BURN + n] - y))
assert err < 1e-10, f"pre-drop rebuild mismatch: {err}"

# ── lag features from s_full, strictly causal: row t sees s_full[< t+BURN] ──
csum = np.concatenate([[0.0], np.cumsum(s_full)])


def ma(r):
    idx = np.arange(n) + BURN
    return (csum[idx] - csum[idx - r]) / r


def kfeat(w):
    """F(t) = sum_u w[u-1] * s[t-u]  for u=1..len(w), exact via fftconvolve."""
    full = fftconvolve(s_full, w, mode="full")[: BURN + n]
    return full[BURN - 1 : BURN + n - 1]  # value at t uses s up to t-1


RUNGS_B5 = [1, 5, 25, 125, 625, 3125]
RUNGS_B2 = [2**k for k in range(12)]
RUNGS_AR = sorted({int(round(v)) for v in np.linspace(1, 2048, 12)})
_r, _v = [], 1.0
while round(_v) <= 2048:
    _r.append(int(round(_v)))
    _v *= np.sqrt(2.0)
RUNGS_SQ2 = sorted(set(_r))
F_B5 = np.column_stack([ma(r) for r in RUNGS_B5])
F_B2 = np.column_stack([ma(r) for r in RUNGS_B2])
F_AR = np.column_stack([ma(r) for r in RUNGS_AR])
F_SQ2 = np.column_stack([ma(r) for r in RUNGS_SQ2])
F_B2PREP = X[:, [names.index(f"har_ma_{r}") for r in RUNGS_B2]]

NB, UMAX = 16, 2048
g = np.log2(np.arange(1, UMAX + 1, dtype=np.float64))
m = NB - 3                                    # internal segments
kn = np.concatenate([[0.0] * 3, np.linspace(0.0, 11.0, m + 1), [11.0] * 3])
BW = BSpline.design_matrix(np.clip(g, 0, 11), kn, 3).toarray()   # (2048, NB)
F_SPL = np.column_stack([kfeat(BW[:, j]) for j in range(NB)])
D2 = np.diff(np.eye(NB), n=2, axis=0)
PEN_SPL = D2.T @ D2

MIX_T = [0.3, 0.6, 1.0, 1.5, 2.2]


def plw(beta, U):
    u = np.arange(1, U + 1, dtype=np.float64)
    w = u ** (-beta)
    return w / w.sum()


def expw(tau, U):
    u = np.arange(1, U + 1, dtype=np.float64)
    w = np.exp(-u / tau)
    return w / w.sum()


F_MIX5 = np.column_stack([kfeat(plw(be, 256)) for be in MIX_T])


def fit_scalar(t, make_w, lo, hi, logspace=False):
    """Best kernel parameter on window [t-W, t) by SSE of OLS y ~ [1, K]."""
    yy = y[t - W : t]

    def loss(v):
        p = float(np.exp(v)) if logspace else float(v)
        K = kfeat(make_w(p))[t - W : t]
        A = np.column_stack([np.ones(W), K])
        cf, *_ = np.linalg.lstsq(A, yy, rcond=None)
        r = yy - A @ cf
        return float(r @ r)

    res = minimize_scalar(loss, bounds=(np.log(lo), np.log(hi)) if logspace
                          else (lo, hi), method="bounded")
    return float(np.exp(res.x)) if logspace else float(res.x)


def run_arm(F_lag, pen, n0, n1, refit=None, standardize=True):
    """Per-bar sliding ridge on [1 | F_lag | CAL] over tile [n0, n1).

    pen: scalar -> alpha*I on lag block; matrix -> generalized penalty
    (spline). Calendar block always alpha=1, intercept unpenalized.
    refit: optional callable t -> F_lag (fitted-kernel arms, per segment).
    Returns preds and the first-window effective df.
    """
    P = np.empty(n1 - n0)
    edf0 = None
    for t0 in range(n0, n1, CADENCE):
        hi = min(t0 + CADENCE, n1)
        FL = refit(t0) if refit is not None else F_lag
        p_lag = FL.shape[1]
        blk = [np.ones((hi - (t0 - W), 1)), FL[t0 - W : hi], CAL[t0 - W : hi]]
        A = np.hstack(blk)
        if standardize:
            mu = A[:W].mean(axis=0)
            sd = A[:W].std(axis=0)
            sd[sd < 1e-12] = 1.0
            mu[0], sd[0] = 0.0, 1.0
            A = (A - mu) / sd
        p = A.shape[1]
        Pm = np.zeros((p, p))
        if np.isscalar(pen):
            Pm[1 : 1 + p_lag, 1 : 1 + p_lag] = pen * np.eye(p_lag)
        else:
            Pm[1 : 1 + p_lag, 1 : 1 + p_lag] = pen
        Pm[1 + p_lag :, 1 + p_lag :] = np.eye(p - 1 - p_lag)
        G = A[:W].T @ A[:W]
        c = A[:W].T @ y[t0 - W : t0]
        if edf0 is None:
            edf0 = float(np.trace(np.linalg.solve(G + Pm, G)))
        for i in range(hi - t0):
            cf = np.linalg.solve(G + Pm, c)
            P[t0 - n0 + i] = A[W + i] @ cf
            if i < hi - t0 - 1:
                anew, aold = A[W + i], A[i]
                G += np.outer(anew, anew) - np.outer(aold, aold)
                c += anew * y[t0 + i] - aold * y[t0 - W + i]
    return P, edf0


def match_lambda(target_edf, n0):
    """Bisect log-lambda so the spline arm's first-window edf == target."""
    A = np.hstack([np.ones((W, 1)), F_SPL[n0 - W : n0], CAL[n0 - W : n0]])
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    mu[0], sd[0] = 0.0, 1.0
    # spline block deliberately NOT rescaled for the penalty's sake in run_arm
    # when pen is a matrix — so mirror that here: standardize only cal+icpt.
    mu[1 : 1 + NB], sd[1 : 1 + NB] = 0.0, 1.0
    A = (A - mu) / sd
    G = A.T @ A

    def edf(loglam):
        Pm = np.zeros((A.shape[1], A.shape[1]))
        Pm[1 : 1 + NB, 1 : 1 + NB] = np.exp(loglam) * PEN_SPL \
            + 1e-8 * np.eye(NB)
        Pm[1 + NB :, 1 + NB :] = np.eye(A.shape[1] - 1 - NB)
        return float(np.trace(np.linalg.solve(G + Pm, G)))

    lo, hi = -10.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if edf(mid) > target_edf:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def spline_pen(lam):
    return lam * PEN_SPL + 1e-8 * np.eye(NB)


def evaluate(tag, Ptile, n0, n1, ref_loss=None):
    pr, tr = apply_duan_smearing(Ptile, y[n0:n1], b[n0:n1])
    mm = forecast_metrics(pr, tr)
    loss = qlike_per_bar(pr, tr)
    np.savez(f"{OUT}/preds_{tag}_t{n0}.npz", pred_raw=pr, true_raw=tr)
    line = f"  {tag:<10} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f}"
    if ref_loss is not None:
        r = dm_test(loss, ref_loss, h=1)
        line += f"  vs b2: diff={r['mean_diff']:+.6f} dm={r['dm']:+.2f} p={r['p']:.4f}"
    print(line, flush=True)
    return loss


TILE_BOUNDS = {1: (24000, 26189), 2: (26189, 28378)}
ALL_ARMS = ["b5", "b2prep", "arith12", "bsq2", "spl", "pl256", "pl2048",
            "exp", "mix5"]
ARMS = [a for a in os.environ.get("ARMS", ",".join(ALL_ARMS)).split(",") if a]

for tile in TILES:
    n0, n1 = TILE_BOUNDS[tile]
    print(f"\n=== tile {tile} [{n0},{n1}) ===", flush=True)
    t0 = time.time()

    P_b2, edf_b2 = run_arm(F_B2, 1.0, n0, n1)
    loss_b2 = evaluate("b2", P_b2, n0, n1)
    print(f"  (b2 edf={edf_b2:.2f})", flush=True)

    for tag, FL in [("b5", F_B5), ("b2prep", F_B2PREP), ("arith12", F_AR),
                    ("bsq2", F_SQ2), ("mix5", F_MIX5)]:
        if tag in ARMS:
            P_, edf_ = run_arm(FL, 1.0, n0, n1)
            evaluate(tag, P_, n0, n1, loss_b2)
            print(f"    (edf={edf_:.2f})", flush=True)

    if "spl" in ARMS:
        lam_star = match_lambda(edf_b2, n0)
        P_sp, edf_sp = run_arm(F_SPL, spline_pen(lam_star), n0, n1,
                               standardize=False)
        print(f"  (spline matched lambda={lam_star:.3g}, edf={edf_sp:.2f})",
              flush=True)
        evaluate("spl", P_sp, n0, n1, loss_b2)
        if os.environ.get("SPL_SWEEP"):
            for lam in [1e-2, 1e0, 1e2, 1e3, 1e4, 1e5, 1e6]:
                P_s, edf_s = run_arm(F_SPL, spline_pen(lam), n0, n1,
                                     standardize=False)
                evaluate(f"spl{lam:g}", P_s, n0, n1, loss_b2)
                print(f"    (edf={edf_s:.2f})", flush=True)

    for U in (256, 2048):
        if f"pl{U}" not in ARMS:
            continue
        cachebeta = {}

        def mk_pl(t, U=U, cache=cachebeta):
            seg = t // CADENCE
            if seg not in cache:
                cache[seg] = fit_scalar(t, lambda be: plw(be, U), 0.05, 3.0)
            return kfeat(plw(cache[seg], U))[:, None]

        P_pl, _ = run_arm(None, 1.0, n0, n1, refit=mk_pl)
        evaluate(f"pl{U}", P_pl, n0, n1, loss_b2)
        print(f"    (beta by segment: "
              f"{[round(v, 3) for v in cachebeta.values()][:5]}...)",
              flush=True)

    if "exp" in ARMS:
        cachetau = {}

        def mk_exp(t, cache=cachetau):
            seg = t // CADENCE
            if seg not in cache:
                cache[seg] = fit_scalar(t, lambda ta: expw(ta, 2048), 1.0,
                                        2048.0, logspace=True)
            return kfeat(expw(cache[seg], 2048))[:, None]

        P_ex, _ = run_arm(None, 1.0, n0, n1, refit=mk_exp)
        evaluate("exp", P_ex, n0, n1, loss_b2)
        print(f"    (tau by segment: "
              f"{[round(v, 1) for v in cachetau.values()][:5]}...)",
              flush=True)

    print(f"  tile {tile} done in {time.time() - t0:.0f}s", flush=True)
