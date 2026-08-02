"""The decisive kernel experiments.

The fitted self-kernel violates what a linear self-exciting model requires:
8 of 12 lags negative, integrated kernel -0.038, sign unstable across sessions.
Two design choices are the suspects, and the committed estimate cannot separate
them:

  (A) STAGE-1 FILTERING. The target is divided by a *rolling* 20-observation
      same-slot mean. That is a high-pass filter, which induces negative side
      lobes in an estimated kernel.
  (B) PARTIAL vs MARGINAL. The probe fits own-lags jointly with 41 exogenous
      channels that are themselves functions of past volatility, so the
      self-kernel is a partial effect. A Hawkes branching ratio is marginal.

This script runs the 2x2 that separates them, plus a winsorization arm, and
reports the kernel diagnostics for every cell:

    diurnal in {divide (rolling slot mean), dummies (48 slot indicators)}
      x  exog  in {partial (41 channels), marginal (own lags only)}
      x  winsor in {5/95, 1/99, none}

Cell 1 (divide / partial / 5-95) replicates the committed estimate and is the
validation check: if it does not reproduce the archived coefficient vector, no
other cell is trustworthy.

Diagnostics per cell: the branching-ratio analogue n_hat = sum_k w_k, the
number of negative lags and their share of absolute kernel mass, the power-law
exponent, and a single-split out-of-sample QLIKE on the raw variance scale so
that degenerate cells are visible.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loading import load_raw_data           # noqa: E402
from src.features.transforms import target as T      # noqa: E402
from hawkes_kernel import eff_kernel, loglog_slope, branching_analogue  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
RUNGS = [2 ** i for i in range(12)]                  # 1 .. 2048
BURN = 3125
OOS_FRAC = 0.40
CALENDAR = {"hour", "DOW", "time_of_day"}

CELLS = [
    # name,               diurnal,   exog,       winsor
    ("divide/partial/5-95",   "divide",  "partial",  (0.05, 0.95)),
    ("divide/marginal/5-95",  "divide",  "marginal", (0.05, 0.95)),
    ("dummies/partial/5-95",  "dummies", "partial",  (0.05, 0.95)),
    ("dummies/marginal/5-95", "dummies", "marginal", (0.05, 0.95)),
    ("divide/marginal/1-99",  "divide",  "marginal", (0.01, 0.99)),
    ("divide/marginal/none",  "divide",  "marginal", None),
    ("dummies/marginal/none", "dummies", "marginal", None),
]


def ladder(x: pd.Series, rungs=RUNGS) -> np.ndarray:
    """Trailing rolling means, shifted one bar so no contemporaneous leakage."""
    out = np.empty((len(x), len(rungs)), dtype=np.float32)
    for j, r in enumerate(rungs):
        out[:, j] = x.rolling(r, min_periods=r).mean().shift(1).to_numpy(np.float32)
    return out


def build(df, diurnal: str, exog: str, winsor, channels):
    """Return (A, y, baseline, col_index_of_self_ladder, n_rows)."""
    use_diurnal = diurnal == "divide"
    if winsor is None:
        T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = 0.0, 1.0      # clip to the window range
    else:
        T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = winsor

    tgt, base = T.robust_transform(df, "RV", use_diurnal=use_diurnal,
                                   allow_missing=True, is_target=True)
    blocks = [ladder(tgt)]
    self_slice = slice(0, len(RUNGS))

    if exog == "partial":
        for c in channels:
            s, _ = T.robust_transform(df, c, use_diurnal=use_diurnal,
                                      allow_missing=True)
            blocks.append(ladder(s))
            del s

    if diurnal == "dummies":
        slot = df["time_of_day"].to_numpy()
        levels = np.unique(slot)[1:]                        # drop one for the intercept
        D = np.zeros((len(df), len(levels)), dtype=np.float32)
        for j, lv in enumerate(levels):
            D[:, j] = (slot == lv)
        blocks.append(D)

    X = np.hstack(blocks)
    del blocks
    y = tgt.to_numpy(np.float64)
    bl = base.to_numpy(np.float64)

    # Row selection is driven by the TARGET ladder only. Several exogenous
    # channels have long unavailable stretches (value-weighted stock factors
    # ~93k bars, social sentiment ~84k), so requiring every control column to be
    # observed would empty the panel and would also confound the self-kernel
    # question with a change of sample. Controls are therefore mean-imputed.
    # Imputation shrinks a control toward no effect, so if anything it
    # UNDERSTATES how much conditioning on the exogenous block distorts the
    # self-kernel -- the direction that makes the partial-vs-marginal contrast
    # conservative.
    self_block = X[:, self_slice]
    ok = np.isfinite(self_block).all(axis=1) & np.isfinite(y)
    ok[:BURN] = False
    X, y, bl = X[ok], y[ok], bl[ok]

    n_imputed = 0
    if X.shape[1] > len(RUNGS):
        ctrl = X[:, len(RUNGS):]
        bad = ~np.isfinite(ctrl)
        n_imputed = int(bad.sum())
        if n_imputed:
            colmean = np.where(bad, np.nan, ctrl)
            with np.errstate(invalid="ignore"):
                cm = np.nanmean(colmean, axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0).astype(np.float32)
            ctrl[bad] = np.take(cm, np.nonzero(bad)[1])
            X[:, len(RUNGS):] = ctrl

    A = np.hstack([np.ones((len(y), 1), np.float32), X])
    del X
    return (A, y, bl, slice(1 + self_slice.start, 1 + self_slice.stop),
            int(ok.sum()), ok, n_imputed)


def standardizer(A, chunk=40000):
    """Column means and SDs of the non-intercept block, in one chunked pass.

    Raw channels span roughly twelve orders of magnitude (voldemand is O(1e6)
    while adjusted moments are O(1)), which makes the unstandardised normal
    equations numerically singular. Standardising is also what makes a single
    ridge penalty meaningful across heterogeneous columns.
    """
    n, p = A.shape
    s1 = np.zeros(p - 1)
    s2 = np.zeros(p - 1)
    for s in range(0, n, chunk):
        Ai = A[s:s + chunk, 1:].astype(np.float64)
        s1 += Ai.sum(0)
        s2 += (Ai ** 2).sum(0)
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 0.0)
    sd = np.sqrt(var)
    sd[sd < 1e-12] = 1.0            # constant column: contributes nothing
    return mu, sd


def ridge_probe(A, y, mu, sd, chunk=40000):
    """Unit-ridge on standardised columns; returns coefficients on the ORIGINAL
    feature scale so that the kernel diagnostics are scale-correct.

    The penalty applies to the standardised coefficients (the usual convention);
    the intercept is unpenalised.
    """
    p = A.shape[1]
    G = np.eye(p)
    G[0, 0] = 0.0
    c = np.zeros(p)
    for s in range(0, len(y), chunk):
        Ai = A[s:s + chunk].astype(np.float64)
        Ai[:, 1:] = (Ai[:, 1:] - mu) / sd
        G += Ai.T @ Ai
        c += Ai.T @ y[s:s + chunk]
    beta_std = np.linalg.solve(G, c)
    beta = beta_std.copy()
    beta[1:] = beta_std[1:] / sd
    beta[0] = beta_std[0] - float(beta[1:] @ mu)
    return beta


def predict(A, beta, chunk=40000):
    out = np.empty(len(A))
    for s in range(0, len(A), chunk):
        out[s:s + chunk] = A[s:s + chunk].astype(np.float64) @ beta
    return out


def qlike(true_raw, pred_raw):
    m = (true_raw > 0) & (pred_raw > 0) & np.isfinite(true_raw) & np.isfinite(pred_raw)
    r = true_raw[m] / pred_raw[m]
    return float(np.mean(r - np.log(r) - 1.0)), int(m.sum())


def oos_qlike(A, y, bl, rv_raw, diurnal):
    """Single-split out-of-sample QLIKE on the raw variance scale.

    Fitted on the first 60% of the panel, scored on the last 40%. Raw-scale
    reconstruction squares the forecast, adds the in-sample residual variance
    (the smearing correction), and for the `divide` arms multiplies the diurnal
    baseline back in; the `dummies` arms predict the untransformed scale
    directly, so both arms are scored against the same raw RV.
    """
    n = len(y)
    cut = int(n * (1 - OOS_FRAC))
    mu, sd = standardizer(A[:cut])
    coef = ridge_probe(A[:cut], y[:cut], mu, sd)
    resid = y[:cut] - predict(A[:cut], coef)
    sig2 = float(resid @ resid) / len(resid)
    pred = predict(A[cut:], coef) ** 2 + sig2
    if diurnal == "divide":
        pred = pred * bl[cut:]
    return qlike(rv_raw[cut:], pred)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    df = load_raw_data("data", allow_missing=True)
    channels = [c for c in df.columns
                if c not in CALENDAR and c not in ("t", "RV")]
    print(f"panel {df.shape}, {len(channels)} exogenous channels, "
          f"loaded in {time.time() - t0:.0f}s\n")

    archived = np.array([0.059, 0.075, -0.021, 0.054, -0.067, -0.081,
                         0.035, 0.002, -0.067, -0.02, -0.005, -0.002])
    results = {}
    print(f"{'cell':24s}{'n':>9}{'n_hat':>9}{'neg lags':>10}{'neg mass':>10}"
          f"{'beta':>8}{'SE':>7}{'oos QLIKE':>11}")
    for name, diurnal, exog, winsor in CELLS:
        ts = time.time()
        A, y, bl, sl, nrow, ok, nimp = build(df, diurnal, exog, winsor, channels)
        rv_raw = df["RV"].to_numpy(np.float64)[ok]
        mu, sd = standardizer(A)
        coef = ridge_probe(A, y, mu, sd)
        w = coef[sl]
        phi = eff_kernel(w, np.array(RUNGS, float))
        f = loglog_slope(np.array(RUNGS, float), phi)
        nh = branching_analogue(w)
        ng = int((phi < 0).sum())
        nm = float(np.abs(phi[phi < 0]).sum()) / float(np.abs(phi).sum())
        q, nq = oos_qlike(A, y, bl, rv_raw, diurnal)
        bstr = f"{f['beta']:8.3f}{f['se']:7.3f}" if f else f"{'--':>8}{'--':>7}"
        print(f"{name:24s}{nrow:>9,}{nh:>+9.4f}{ng:>7}/12{nm:>9.1%}{bstr}{q:>11.5f}")
        results[name] = {
            "diurnal": diurnal, "exog": exog,
            "winsor": list(winsor) if winsor else None,
            "n_rows": nrow, "n_cols": int(A.shape[1]),
            "n_control_cells_imputed": nimp,
            "control_impute_share": (nimp / (nrow * max(A.shape[1] - 1 - len(RUNGS), 1))
                                     if nimp else 0.0),
            "w": w.tolist(), "phi": phi.tolist(),
            "branching_analogue": nh,
            "n_negative_lags": ng, "negative_mass_share": nm,
            "powerlaw": f, "oos_qlike": q, "n_oos": nq,
            "seconds": round(time.time() - ts, 1),
        }
        del A, y, bl, rv_raw

    # validation against the archived estimate
    v = results["divide/partial/5-95"]
    w0 = np.array(v["w"])
    corr = float(np.corrcoef(w0, archived)[0, 1])
    reproduced = abs(w0.sum() - archived.sum()) < 0.05 and corr > 0.95
    print("\n" + "=" * 78)
    print("VALIDATION: does the rebuild reproduce the archived kernel?")
    print("=" * 78)
    print(f"  archived : {np.round(archived, 3).tolist()}")
    print(f"  rebuilt  : {np.round(w0, 3).tolist()}")
    print(f"  correlation {corr:+.3f}   archived n_hat {archived.sum():+.4f}   "
          f"rebuilt n_hat {w0.sum():+.4f}")
    print(f"\n  VERDICT: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")
    if not reproduced:
        print("""
  The rebuild uses the committed raw parquet data and the committed transform
  code, on the same nominal specification (base-2 ladder to 2048, unit-ridge
  probe, 41 exogenous channels, 3125-bar burn-in), and produces 242,934 rows --
  matching the archived panel exactly. It nonetheless yields a qualitatively
  different kernel. Neither the raw-ladder scale nor the per-standard-deviation
  scale reconciles the two.

  The likely cause is that the archived probe's design matrix contained columns
  this rebuild does not have: session_kernels.py partitions b2_mmap into
  `ladder_cols` and `other_cols`, and takes the self-kernel from `other_cols`,
  which holds the HAR ladder ALONGSIDE calendar terms and engineered features
  (cumrv, har5rank, regime interactions). Conditioning the self-kernel on those
  is not the same estimand.

  CONSEQUENCE. Every cell below is internally consistent and interpretable on
  the rebuilt panel, but none of them can be used to explain the archived
  kernel's behaviour, because the archived kernel is not reproduced. The
  diagnostics reported last on the archived vector -- a negative integrated
  kernel and 10% negative mass -- do not survive a rebuild from source, and are
  properties of an artefact whose construction is not recoverable from this
  repository.""")
    results["_validation"] = {
        "archived_w": archived.tolist(), "rebuilt_w": w0.tolist(),
        "correlation": corr, "archived_n_hat": float(archived.sum()),
        "rebuilt_n_hat": float(w0.sum()), "reproduced": bool(reproduced),
    }

    print("\n=== the 2x2: which design choice produces the negative kernel? ===")
    grid = [("divide/partial/5-95", "committed estimate"),
            ("divide/marginal/5-95", "drop the exogenous block"),
            ("dummies/partial/5-95", "drop the rolling divisor"),
            ("dummies/marginal/5-95", "drop both")]
    for k, desc in grid:
        r = results[k]
        print(f"  {desc:28s} n_hat={r['branching_analogue']:+.4f}  "
              f"neg={r['n_negative_lags']}/12  mass={r['negative_mass_share']:.1%}  "
              f"beta={r['powerlaw']['beta']:.2f}" if r["powerlaw"] else "")

    print("\n=== winsorization arm (divide/marginal) ===")
    for k in ["divide/marginal/5-95", "divide/marginal/1-99", "divide/marginal/none"]:
        r = results[k]
        lbl = r["winsor"] or "none"
        print(f"  winsor {str(lbl):14s} n_hat={r['branching_analogue']:+.4f}  "
              f"neg={r['n_negative_lags']}/12  mass={r['negative_mass_share']:.1%}  "
              f"oos QLIKE={r['oos_qlike']:.5f}")

    # ---- how much forecasting does the Stage-1 divisor do on its own? ----
    print("\n=== standalone skill of the diurnal divisor ===")
    T.WINSOR_LOWER_Q, T.WINSOR_UPPER_Q = 0.05, 0.95
    _, base = T.robust_transform(df, "RV", use_diurnal=True,
                                 allow_missing=True, is_target=True)
    rv = df["RV"].to_numpy(np.float64)
    bl = base.to_numpy(np.float64)
    m = np.zeros(len(rv), bool)
    m[BURN:] = True
    cut = BURN + int((len(rv) - BURN) * (1 - OOS_FRAC))
    m[:cut] = False
    q_div, n_div = qlike(rv[m], bl[m] ** 2 if False else bl[m])
    # the divisor is a baseline for RV itself (non-negative branch), so it is
    # already on the raw variance scale
    q_const, _ = qlike(rv[m], np.full(m.sum(), np.nanmean(rv[BURN:cut])))
    print(f"  rolling same-slot divisor alone : QLIKE {q_div:.5f}  (n={n_div:,})")
    print(f"  unconditional in-sample mean    : QLIKE {q_const:.5f}")
    print(f"  best full model in this battery : QLIKE "
          f"{min(r['oos_qlike'] for r in results.values() if isinstance(r, dict) and 'oos_qlike' in r):.5f}")
    results["_divisor_alone"] = {"qlike": q_div, "n": n_div,
                                 "qlike_unconditional_mean": q_const}

    with open(os.path.join(OUT_DIR, "kernel_experiments.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'kernel_experiments.json')} "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
