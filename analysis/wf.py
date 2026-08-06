"""Walk-forward helpers shared by the alpha-manifestation study.

Thin wrappers over :class:`src.models.rolling_least_squares.RollingLeastSquares` — the
same rank-1 sliding-window ridge the production Ridge path uses — so every number in the
study comes from a strictly causal, refit-as-you-go fit rather than a full-sample one.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rolling_least_squares import RollingLeastSquares  # noqa: E402


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    alpha: float = 1.0,
    refit_every: int = 1,
) -> np.ndarray:
    """Rolling-window ridge OOS predictions for rows ``[train_win, n)``.

    Returns an array of length ``n - train_win``: prediction for row
    ``train_win + i`` fit on the ``train_win`` rows strictly before it.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(X)
    solver = RollingLeastSquares(alpha=alpha, fit_intercept=True)
    solver.init_window(X[:train_win], y[:train_win])
    solver.solve()
    out = np.empty(n - train_win, dtype=np.float64)
    for i in range(n - train_win):
        t = train_win + i
        if i and i % refit_every == 0:
            solver.solve()
        out[i] = solver.predict_one(X[t])
        solver.roll(X[t], y[t], X[t - train_win], y[t - train_win])
    return out


def walk_forward_coefs(
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    alpha: float = 1.0,
    every: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficient path: ``(row_index, coefs)`` sampled every ``every`` rows."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(X)
    solver = RollingLeastSquares(alpha=alpha, fit_intercept=True)
    solver.init_window(X[:train_win], y[:train_win])
    rows, coefs = [], []
    for i in range(n - train_win):
        t = train_win + i
        if i % every == 0:
            solver.solve()
            rows.append(t)
            coefs.append(solver._coef.copy())
        solver.roll(X[t], y[t], X[t - train_win], y[t - train_win])
    return np.asarray(rows), np.asarray(coefs)


def nw_tstat(x: np.ndarray, y: np.ndarray, lags: int = 48) -> tuple[float, float]:
    """(slope, Newey-West t) of ``y ~ a + b x`` with ``lags`` Bartlett lags.

    Both series are bar-level and heavily autocorrelated; an OLS t on 200k bars is
    meaningless, so every reported t in the study is HAC.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return 0.0, 0.0
    b = float(xc @ (y - y.mean())) / sxx
    u = (y - y.mean()) - b * xc
    g = xc * u
    s = float(g @ g)
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * float(g[L:] @ g[:-L])
    se = np.sqrt(max(s, 0.0)) / sxx
    return b, (b / se if se > 0 else 0.0)


def r2_oos(y: np.ndarray, pred: np.ndarray) -> float:
    """OOS R^2 against the *mean-zero* benchmark (the residual's own null)."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return 1.0 - float(((y - pred) ** 2).sum()) / float((y**2).sum())


def walk_forward_embargo_blocked(
    X: np.ndarray,
    y: np.ndarray,
    day_codes: np.ndarray,
    train_days: int = 250,
    embargo_days: int = 1,
    alpha: float = 1.0,
) -> np.ndarray:
    """Day-blocked embargoed rolling ridge — the msweep block_ridge lesson as a shared helper.

    Same object as ``walk_forward_embargo`` at daily refit, day-quantized: predictions for every
    bar of day d use one ridge solved on the whole-day window [d - embargo_days - train_days,
    d - embargo_days). One BLAS gram update per day instead of per-bar rank-1 rolls (each day's
    gram is computed twice — on entry and on exit — keeping memory at O(p^2)). Embargo is in
    DAYS, rounded UP from the label horizon: stricter than the bar-level embargo, never looser.
    Intercept unpenalized. Returns full-length predictions (NaN in warmup). Engines differ at the
    third decimal on window boundaries — use the SAME engine for both arms of any comparison.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.nan_to_num(np.ascontiguousarray(y, dtype=np.float64))
    n, p = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    starts = np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1]])
    bounds = np.r_[starts, n]
    ndays = len(starts)

    def gram(di: int):
        s, t = bounds[di], bounds[di + 1]
        Zi = Xa[s:t]
        return Zi.T @ Zi, Zi.T @ y[s:t]

    G = np.zeros((p + 1, p + 1))
    bv = np.zeros(p + 1)
    reg = np.diag(np.r_[np.full(p, alpha), 0.0])
    out = np.full(n, np.nan)
    added = 0
    lo = 0
    for d in range(train_days + embargo_days, ndays):
        hi = d - embargo_days
        while added < hi:
            g, b = gram(added)
            G += g
            bv += b
            added += 1
        while added - lo > train_days:
            g, b = gram(lo)
            G -= g
            bv -= b
            lo += 1
        try:
            beta = np.linalg.solve(G + reg, bv)
        except np.linalg.LinAlgError:
            continue
        s, t = bounds[d], bounds[d + 1]
        out[s:t] = Xa[s:t] @ beta
    return out


def walk_forward_embargo_dualcadence(
    X: np.ndarray,
    y: np.ndarray,
    train_win: int,
    embargo: int,
    alpha: float,
    periods_per_day: int = 48,
    rebuild_every: int = 1008,
    fresh_mask: np.ndarray | None = None,
):
    """Per-bar AND daily-refit predictions in ONE pass — the §29 cadence question without
    paying per-bar solves. Maintains P = (G + reg)^-1 by Sherman-Morrison rank-1 updates as the
    bar-quantized window [t-embargo-W, t-embargo) slides; per-bar beta = P @ c every bar, the
    daily arm snapshots beta at each day boundary. Exact inverse rebuild every ``rebuild_every``
    rolls bounds FP drift. Both arms share identical window mechanics, so the cadence contrast
    is engine-clean. Returns (perbar, daily), each aligned like ``walk_forward``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.nan_to_num(np.ascontiguousarray(y, dtype=np.float64))
    n, p = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    reg = np.r_[np.full(p, alpha), 0.0]
    G = Xa[:train_win].T @ Xa[:train_win] + np.diag(reg)
    c = Xa[:train_win].T @ y[:train_win]
    P = np.linalg.inv(G)
    perbar = np.full(n - train_win, np.nan)
    daily = np.full(n - train_win, np.nan)
    hyb_fresh = hyb_rest = None
    fm = None
    if fresh_mask is not None:
        fm = np.r_[np.asarray(fresh_mask, bool), False]  # intercept rides the daily group
        hyb_fresh = np.full(n - train_win, np.nan)  # fresh block per-bar, rest daily
        hyb_rest = np.full(n - train_win, np.nan)   # rest per-bar, fresh block daily
    beta_day = P @ c
    for step, t in enumerate(range(train_win + embargo, n)):
        j = t - embargo
        xin, xout = Xa[j], Xa[j - train_win]
        # add row j
        Pu = P @ xin
        P -= np.outer(Pu, Pu) / (1.0 + xin @ Pu)
        c += xin * y[j]
        # remove row j - train_win
        Pu = P @ xout
        P += np.outer(Pu, Pu) / (1.0 - xout @ Pu)
        c -= xout * y[j - train_win]
        if step and step % rebuild_every == 0:
            Zw = Xa[j - train_win + 1 : j + 1]
            P = np.linalg.inv(Zw.T @ Zw + np.diag(reg))
        beta = P @ c
        if step % periods_per_day == 0:
            beta_day = beta
        perbar[t - train_win] = Xa[t] @ beta
        daily[t - train_win] = Xa[t] @ beta_day
        if fm is not None:
            xb = Xa[t]
            hyb_fresh[t - train_win] = xb[fm] @ beta[fm] + xb[~fm] @ beta_day[~fm]
            hyb_rest[t - train_win] = xb[~fm] @ beta[~fm] + xb[fm] @ beta_day[fm]
    if fm is not None:
        return perbar, daily, hyb_fresh, hyb_rest
    return perbar, daily
