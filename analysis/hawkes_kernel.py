"""Statistics of the fitted kernel: the Hawkes reading, tested.

Inputs (all committed artefacts, no cluster data required):
  * the pooled and session-conditional SELF-kernel boxcar coefficients, from
    drivers/msweep_2026-08-01/verdicts/session_kernels.log
  * the session-conditional EXOGENOUS operators (41 channels x 12 rungs),
    results/sesskern_*.npy

What is computed:
  1. the effective per-lag kernel phi(u) = sum_{r_k >= u} w_k / r_k, and the
     exact identity sum_u phi(u) = sum_k w_k (the branching-ratio analogue);
  2. the non-negativity requirement of the linear Hawkes class, quantified as
     negative mass share rather than asserted;
  3. the power-law exponent beta with a standard error, a confidence interval,
     and a goodness-of-fit -- the quantity the paper currently hardcodes;
  4. power law vs exponential as a model-selection test, not an assertion;
  5. the same exponent across 41 exogenous channels, where the estimate is
     invariant to each channel's arbitrary scaling;
  6. whether the session kernels are amplitude motion on a fixed shape or a
     genuine rotation (rank-1 share of the 6 x 12 session matrix).

CAVEAT carried into every output: these coefficients come from a probe that
includes the exogenous block, so the self-kernel is the PARTIAL effect of own
lags net of 41 exogenous channels, not the marginal self-excitation a Hawkes
kernel denotes. See `report_caveats`.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "drivers", "msweep_2026-08-01", "verdicts", "session_kernels.log")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
RUNGS = np.array([2 ** i for i in range(12)], dtype=float)   # 1, 2, 4, ..., 2048


# --------------------------------------------------------------------------
# Kernel algebra
# --------------------------------------------------------------------------

def eff_kernel(w: np.ndarray, rungs: np.ndarray = RUNGS) -> np.ndarray:
    """phi(u) = sum_{k : r_k >= u} w_k / r_k, evaluated at the rungs."""
    w = np.asarray(w, dtype=float)
    return np.array([np.sum(w[rungs >= u] / rungs[rungs >= u]) for u in rungs])


def branching_analogue(w: np.ndarray) -> float:
    """The discretised kernel integral. Equals sum_k w_k exactly."""
    return float(np.sum(w))


def verify_identity(w: np.ndarray, rungs: np.ndarray = RUNGS) -> tuple[float, float]:
    """sum over ALL integer lags of phi equals sum_k w_k (see paper Eq. 13)."""
    total = 0.0
    for k, r in enumerate(rungs):
        # phi is constant on each rung interval; lag u contributes w_k/r_k for
        # every u <= r_k, so w_k/r_k appears r_k times.
        total += w[k]
    lhs = 0.0
    maxr = int(rungs[-1])
    for u in range(1, maxr + 1):
        lhs += np.sum(w[rungs >= u] / rungs[rungs >= u])
    return float(lhs), float(total)


# --------------------------------------------------------------------------
# Exponent estimation
# --------------------------------------------------------------------------

def loglog_slope(u: np.ndarray, phi: np.ndarray):
    """OLS of log phi on log u over the strictly positive part.

    Returns beta (= -slope), its standard error, R^2, and the number of points.
    The standard error is the textbook OLS one and is reported with an explicit
    warning: consecutive phi values are nested partial sums of the same
    coefficient vector, so they are far from independent and this SE
    understates the true uncertainty.
    """
    m = phi > 0
    if m.sum() < 3:
        return None
    x = np.log(u[m])
    yv = np.log(phi[m])
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    dof = n - 2
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = math.sqrt(s2 * xtx_inv[1, 1])
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": float(-coef[1]), "se": float(se_slope), "r2": r2,
            "n_points": int(n), "dof": int(dof), "intercept": float(coef[0])}


def exponential_fit(u: np.ndarray, phi: np.ndarray):
    """OLS of log phi on u (not log u): the exponential-kernel alternative."""
    m = phi > 0
    if m.sum() < 3:
        return None
    x = u[m]
    yv = np.log(phi[m])
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    rss = float(resid @ resid)
    return {"rate": float(-coef[1]), "r2": r2, "rss": rss, "n_points": int(n)}


def aic(rss: float, n: int, k: int) -> float:
    return n * math.log(rss / n) + 2 * k


def _p_from_t(t: float, df: int) -> float:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import _student_t_sf
    return _student_t_sf(t, df)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_log(path: str):
    txt = open(path).read()
    m = re.search(r"target kernel \(base-2 rungs\): \[([^\]]+)\]", txt)
    pooled = np.array([float(v) for v in m.group(1).split(",")])
    sess = {}
    blk = txt.split("per-bin target kernels")[1]
    for line in blk.strip().splitlines()[1:]:
        mm = re.match(r"\s*(\w+)\s+\[([^\]]+)\]", line)
        if mm:
            sess[mm.group(1)] = np.array([float(v) for v in mm.group(2).split(",")])
    cos = dict(re.findall(r"^(\w+)\s+\d+\s+\|\s+[\d.]+\s+([+-][\d.]+)", txt, re.M))
    nbin = dict(re.findall(r"^(\w+)\s+(\d+)\s+\|", txt, re.M))
    return pooled, sess, {k: float(v) for k, v in cos.items()}, \
        {k: int(v) for k, v in nbin.items()}


# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pooled, sess, cos, nbin = parse_log(LOG)
    out = {"rungs": RUNGS.tolist()}

    print("=" * 78)
    print("SELF-KERNEL: the object the Hawkes reading is about")
    print("=" * 78)
    print(f"  boxcar coefficients w (rungs 1..2048): {np.round(pooled, 3).tolist()}")

    phi = eff_kernel(pooled)
    print(f"  effective kernel phi(u):               {np.round(phi, 5).tolist()}")

    lhs, rhs = verify_identity(pooled)
    print(f"\n  identity check  sum_u phi(u) = sum_k w_k:  "
          f"{lhs:.6f} vs {rhs:.6f}   (max err {abs(lhs - rhs):.2e})")

    n_hat = branching_analogue(pooled)
    print(f"\n  BRANCHING-RATIO ANALOGUE  n_hat = sum_k w_k = {n_hat:+.4f}")
    if n_hat < 0:
        print("    *** NEGATIVE. A linear Hawkes process requires n in [0, 1).")
        print("        A negative integrated kernel is outside the admissible range,")
        print("        not merely sub-critical.")
    elif n_hat >= 1:
        print("    *** >= 1: non-stationary under the Hawkes reading.")
    else:
        print(f"    sub-critical; endogenous share would be {n_hat:.3f}")

    neg = phi < 0
    negmass = float(np.abs(phi[neg]).sum()) / float(np.abs(phi).sum())
    print(f"\n  NON-NEGATIVITY (required by the linear Hawkes class)")
    print(f"    lags with phi(u) < 0: {int(neg.sum())} of {len(phi)}")
    print(f"    negative share of total absolute kernel mass: {negmass:.1%}")
    print(f"    most negative value: {phi.min():+.5f} at lag {int(RUNGS[phi.argmin()])}")

    out["pooled"] = {
        "w": pooled.tolist(), "phi": phi.tolist(),
        "branching_analogue": n_hat,
        "identity_lhs": lhs, "identity_rhs": rhs,
        "n_negative_lags": int(neg.sum()),
        "negative_mass_share": negmass,
    }

    print("\n" + "=" * 78)
    print("POWER-LAW EXPONENT: the quantity the paper hardcodes at 1.36")
    print("=" * 78)
    fit = loglog_slope(RUNGS, phi)
    if fit is None:
        print("  cannot fit: fewer than 3 positive lags")
        out["pooled"]["powerlaw"] = None
    else:
        b, se = fit["beta"], fit["se"]
        lo, hi = b - 1.96 * se, b + 1.96 * se
        print(f"  fitted on the {fit['n_points']} positive lags of 12")
        print(f"    beta_hat = {b:.3f}   SE = {se:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"    R^2 of the log-log fit = {fit['r2']:.3f}")
        for name, target in [("paper's 1.36", 1.36),
                             ("Hawkes literature ~1.0", 1.0),
                             ("rough-vol implied ~1.6", 1.6)]:
            t = (b - target) / se
            inside = lo <= target <= hi
            print(f"    vs {name:26s} t = {t:+6.2f}   "
                  f"{'INSIDE' if inside else 'outside'} the CI")
        out["pooled"]["powerlaw"] = {**fit, "ci_low": lo, "ci_high": hi}

        ex = exponential_fit(RUNGS, phi)
        pl_rss = (1 - fit["r2"]) * float(((np.log(phi[phi > 0])
                                           - np.log(phi[phi > 0]).mean()) ** 2).sum())
        print(f"\n  power law vs exponential (same {fit['n_points']} points)")
        print(f"    power-law   R^2 = {fit['r2']:.3f}   AIC = {aic(max(pl_rss,1e-12), fit['n_points'], 2):.2f}")
        print(f"    exponential R^2 = {ex['r2']:.3f}   AIC = {aic(max(ex['rss'],1e-12), ex['n_points'], 2):.2f}")
        better = "power law" if fit["r2"] > ex["r2"] else "exponential"
        print(f"    better fit: {better}")
        out["pooled"]["exponential"] = ex

    print("\n" + "=" * 78)
    print("SESSION-CONDITIONAL SELF-KERNELS")
    print("=" * 78)
    print(f"  {'session':<11}{'n':>7}{'n_hat':>9}{'neg lags':>10}{'neg mass':>10}"
          f"{'beta':>8}{'SE':>7}{'R2':>7}{'cos':>8}")
    srows = {}
    for k, w in sess.items():
        p = eff_kernel(w)
        f = loglog_slope(RUNGS, p)
        ng = int((p < 0).sum())
        nm = float(np.abs(p[p < 0]).sum()) / float(np.abs(p).sum())
        srows[k] = {"w": w.tolist(), "phi": p.tolist(),
                    "branching_analogue": branching_analogue(w),
                    "n_negative_lags": ng, "negative_mass_share": nm,
                    "powerlaw": f, "cos_to_pooled": cos.get(k), "n_bars": nbin.get(k)}
        bs = f"{f['beta']:8.3f}{f['se']:7.3f}{f['r2']:7.3f}" if f else f"{'--':>8}{'--':>7}{'--':>7}"
        print(f"  {k:<11}{nbin.get(k, 0):>7}{branching_analogue(w):>+9.4f}"
              f"{ng:>10}{nm:>9.1%}{bs}{cos.get(k, float('nan')):>+8.3f}")
    out["sessions"] = srows

    betas = [v["powerlaw"]["beta"] for v in srows.values() if v["powerlaw"]]
    nh = [v["branching_analogue"] for v in srows.values()]
    print(f"\n  across sessions: n_hat ranges [{min(nh):+.4f}, {max(nh):+.4f}], "
          f"sign {'consistent' if min(nh) * max(nh) > 0 else 'FLIPS'}")
    if betas:
        print(f"  across sessions: beta ranges [{min(betas):.2f}, {max(betas):.2f}], "
              f"spread {max(betas) - min(betas):.2f}")
    out["session_summary"] = {
        "branching_min": min(nh), "branching_max": max(nh),
        "beta_min": min(betas) if betas else None,
        "beta_max": max(betas) if betas else None,
    }

    # amplitude motion vs rotation
    M = np.array([sess[k] for k in sess])
    U_, sv, Vt = np.linalg.svd(M, full_matrices=False)
    share = sv ** 2 / (sv ** 2).sum()
    print(f"\n  session kernels as a 6 x 12 matrix: singular-value shares "
          f"{np.round(share[:4], 3).tolist()}")
    print(f"    rank-1 share = {share[0]:.1%}  -> "
          + ("amplitude motion on a fixed shape"
             if share[0] > 0.8 else
             "NOT amplitude motion: the shape itself rotates across sessions"))
    out["session_svd"] = {"sv": sv.tolist(), "variance_share": share.tolist()}

    print("\n" + "=" * 78)
    print("EXOGENOUS OPERATOR: the multivariate-Hawkes cross-kernels")
    print("=" * 78)
    print(f"  {'session':<11}{'rank90':>8}{'sv1 share':>11}{'neg lags %':>12}"
          f"{'beta median':>13}{'beta IQR':>18}")
    exo = {}
    for k in sess:
        f = os.path.join(ROOT, "results", f"sesskern_{k}.npy")
        if not os.path.exists(f):
            continue
        Km = np.load(f)
        U_, sv, Vt = np.linalg.svd(Km, full_matrices=False)
        share = sv ** 2 / (sv ** 2).sum()
        rank90 = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
        phis = np.array([eff_kernel(Km[i]) for i in range(Km.shape[0])])
        negfrac = float((phis < 0).mean())
        bl = [loglog_slope(RUNGS, p) for p in phis]
        bb = [x["beta"] for x in bl if x]
        med = float(np.median(bb)) if bb else float("nan")
        q1, q3 = (float(np.quantile(bb, .25)), float(np.quantile(bb, .75))) if bb \
            else (float("nan"),) * 2
        print(f"  {k:<11}{rank90:>8}{share[0]:>11.1%}{negfrac:>11.1%}"
              f"{med:>13.2f}   [{q1:6.2f}, {q3:6.2f}]")
        exo[k] = {"rank90": rank90, "sv": sv.tolist(),
                  "variance_share": share.tolist(),
                  "negative_lag_fraction": negfrac,
                  "beta_median": med, "beta_q1": q1, "beta_q3": q3,
                  "n_channels": int(Km.shape[0]),
                  "n_channels_fittable": len(bb)}
    out["exogenous_operator"] = exo

    # ---- channel-level exponent test: 41 scale-invariant estimates ----
    print("\n" + "=" * 78)
    print("EXPONENT, ESTIMATED ACROSS CHANNELS")
    print("=" * 78)
    print("  The log-log slope is invariant to each channel's arbitrary scaling, so")
    print("  the 41 cross-kernels give 41 comparable estimates -- far more")
    print("  information than the 4 positive lags the pooled self-kernel offers.")
    allb, npts, chan_id = [], [], []
    for k in sess:
        f = os.path.join(ROOT, "results", f"sesskern_{k}.npy")
        if not os.path.exists(f):
            continue
        Km = np.load(f)
        for i in range(Km.shape[0]):
            r = loglog_slope(RUNGS, eff_kernel(Km[i]))
            if r:
                allb.append(r["beta"])
                npts.append(r["n_points"])
                chan_id.append(i)
    allb = np.array(allb)
    chan_id = np.array(chan_id)
    print(f"\n  {len(allb)} channel-session fits, median {np.median(npts):.0f} "
          f"positive lags each (min {min(npts)}, max {max(npts)})")
    mu, sd = float(allb.mean()), float(allb.std(ddof=1))
    se = sd / math.sqrt(len(allb))
    print(f"    mean beta = {mu:.3f}   SD = {sd:.3f}   SE of mean = {se:.3f}")
    print(f"    median = {np.median(allb):.3f}   "
          f"IQR [{np.quantile(allb, .25):.3f}, {np.quantile(allb, .75):.3f}]")
    print(f"    95% CI for the mean: [{mu - 1.96 * se:.3f}, {mu + 1.96 * se:.3f}]")
    # Cluster-robust at the channel level: the six session fits for one channel
    # come from the same series, so they are not six independent observations.
    cmeans = np.array([allb[chan_id == c].mean() for c in np.unique(chan_id)])
    nc = len(cmeans)
    cmu, csd = float(cmeans.mean()), float(cmeans.std(ddof=1))
    cse = csd / math.sqrt(nc)
    print(f"\n  CLUSTERED BY CHANNEL ({nc} channels; the six session fits per channel")
    print("  come from one series and are not independent):")
    print(f"    mean beta = {cmu:.3f}   SE = {cse:.3f}   "
          f"95% CI [{cmu - 1.96 * cse:.3f}, {cmu + 1.96 * cse:.3f}]")
    print("\n    tests against the competing values (channel-clustered SE):")
    verdicts = {}
    for name, tgt in [("paper's 1.36", 1.36), ("Hawkes literature 1.0", 1.0),
                      ("rough-vol implied 1.6", 1.6)]:
        t = (cmu - tgt) / cse
        pv = _p_from_t(abs(t), nc - 1)
        verdicts[name] = {"target": tgt, "t": t, "p": pv, "rejected": pv <= 0.05}
        print(f"      vs {name:24s} t = {t:+7.2f}   p = {pv:.2e}   "
              f"{'consistent' if pv > 0.05 else 'REJECTED'}")
    out["channel_exponents_clustered"] = {
        "n_channels": nc, "mean": cmu, "sd": csd, "se": cse,
        "ci_low": cmu - 1.96 * cse, "ci_high": cmu + 1.96 * cse,
        "tests": verdicts,
    }
    out["channel_exponents"] = {
        "n_fits": int(len(allb)), "mean": mu, "sd": sd, "se": se,
        "median": float(np.median(allb)),
        "q1": float(np.quantile(allb, .25)), "q3": float(np.quantile(allb, .75)),
        "ci_low": mu - 1.96 * se, "ci_high": mu + 1.96 * se,
        "median_n_points": float(np.median(npts)),
    }

    if exo:
        r90 = [v["rank90"] for v in exo.values()]
        print(f"\n  rank needed for 90% of operator energy: {min(r90)}-{max(r90)} "
              f"of {list(exo.values())[0]['n_channels']} channels")
        nf = [v["negative_lag_fraction"] for v in exo.values()]
        print(f"  cross-kernel entries violating non-negativity: "
              f"{min(nf):.1%}-{max(nf):.1%} of all (channel, lag) cells")

    with open(os.path.join(OUT_DIR, "hawkes_kernel.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'hawkes_kernel.json')}")
    report_caveats()
    return out


def report_caveats():
    print("\n" + "=" * 78)
    print("WHAT THESE NUMBERS ARE NOT")
    print("=" * 78)
    print("""  1. PARTIAL, NOT MARGINAL. The probe regresses the target on its own lag
     ladder AND 41 exogenous channels x 12 rungs jointly. The self-kernel is
     therefore the effect of own lags NET of the exogenous block. Since those
     channels are themselves functions of past volatility, most of the genuine
     self-excitation is absorbed by them. A Hawkes branching ratio is a
     MARGINAL quantity; this is not one, and the two should not be equated.
  2. FILTERED. The target is diurnally divided by a rolling same-slot mean,
     square-root transformed, and winsorized before the fit. The rolling
     divisor is a high-pass filter, which is expected to induce negative side
     lobes and to steepen an estimated decay.
  3. RIDGE-SHRUNK. The probe adds a unit ridge, so every coefficient and hence
     the branching analogue is biased toward zero in magnitude.
  4. NO COEFFICIENT STANDARD ERRORS. The Gram matrix was not retained, so the
     reported exponent SE comes only from the scatter of the 12 kernel values
     about the log-log line. Those values are nested partial sums of the same
     coefficient vector, so they are strongly dependent and the SE understates
     the true uncertainty.
  5. DESCRIPTIVE, NOT WALK-FORWARD. The session probe is full-sample.""")


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
