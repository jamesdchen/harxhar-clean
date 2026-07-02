"""FREE PILOT (no new data): does the options-vol-demand channel carry DIRECTIONAL content?

Before spending on a full SPX option chain to build real dealer-GEX features, test the cheap proxies we
already have -- the `voldemand_*` family (net vol demand = a coarse read on customer options flow, i.e.
the sign that sets dealer gamma) and the VIX term structure -- for signed-return predictability. Two
hypotheses, both through the SAME gate that killed our vol-forecast levers (purged walk-forward +
circular-shift placebo + replication, reused verbatim from src.evaluation.feature_cv); only the objective
is swapped QLIKE -> forward-return MSE (this is a DIRECTION problem, not a variance problem):

  H1 flow_levels : voldemand + VIX-term LEVELS predict next-bar signed return (direct directional content).
  H2 flow_x_mom  : recent-return x voldemand INTERACTION predicts it -- THE gamma-regime mechanism
                   (customer vol demand flips dealer gamma sign -> flips the sign of return autocorrelation:
                   high demand => dealers short gamma => momentum; low demand => long gamma => reversion).

Base = short-horizon momentum lags (recent cumulative signed returns). Incremental value of flow OVER
momentum is the test. Target = next SESSION bar's sumret (intraday only: t and t+1 same date & in 9-15),
fabricated (ffill-0) bars excluded. Causal throughout: all features known at end of bar t, target at t+1.

Reports the gate verdict + interpretable effect size (OOS corr, HAC cov-t, net-of-1-tick sign Sharpe).
Writes results/flow_dir_pilot/summary.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.evaluation.feature_cv import purged_walk_forward, significance_gate
from vrp_screen_worker import build

OUT = "results/flow_dir_pilot"
ALPHA = 10.0
TICK_COST = 0.25 / 6000.0
MOM_LAGS = (1, 3, 6, 13)  # ~half-hour .. ~full session, in 30-min bars


def zscore_causal(x: np.ndarray) -> np.ndarray:
    """Expanding causal standardization (mean/std from strictly past), NaN->0 pre-warmup. No fixed clip."""
    s = pd.Series(x)
    mu = s.expanding(min_periods=250).mean().shift(1)
    sd = s.expanding(min_periods=250).std().shift(1)
    z = (s - mu) / sd.where(sd > 0)
    return np.nan_to_num(z.to_numpy(), nan=0.0)


def mom_features(r: np.ndarray) -> np.ndarray:
    """Recent cumulative signed returns known at end of bar t (causal); shift(1)-free because r[t] is in
    the information set when predicting r[t+1]."""
    s = pd.Series(r)
    return np.column_stack(
        [s.rolling(L, min_periods=1).sum().to_numpy() for L in MOM_LAGS]
    )


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(
        sd < 1e-12, 1.0, sd
    )  # numerical guard for constant column, not a signal clip
    Xs = (X - mu) / sd
    p = Xs.shape[1]
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(p), Xs.T @ (y - y.mean()))
    return mu, sd, beta, float(y.mean())


def predict_ridge(model, X):
    mu, sd, beta, b0 = model
    return ((X - mu) / sd) @ beta + b0


def cov_t_hac(f: np.ndarray, r: np.ndarray, L: int = 5) -> float:
    z = f * r - (f * r).mean()
    gam = [float((z[k:] * z[: len(z) - k]).mean()) for k in range(L + 1)]
    lrv = gam[0] + 2 * sum((1 - k / (L + 1)) * gam[k] for k in range(1, L + 1))
    return float((f * r).mean() / np.sqrt(lrv / len(z) + 1e-300))


def score_dir(Xbase, cand, y, folds, *, n_boot=8, seed0=0):
    """Directional analog of feature_cv.score_feature: delta = fwd-return MSE(base+cand) - MSE(base),
    per (fold, bootstrap); circular-shift placebo on the candidate block. Negative delta => cand helps."""
    cand = cand.reshape(len(cand), -1)
    Xf = np.column_stack([Xbase, cand])
    rng = np.random.default_rng(seed0)
    deltas, plac, fold_means = [], [], []
    for tr, te in folds:
        fold_d = []
        for s in range(n_boot):
            tri = tr[rng.integers(0, len(tr), len(tr))]
            p0 = predict_ridge(fit_ridge(Xbase[tri], y[tri], ALPHA), Xbase[te])
            p1 = predict_ridge(fit_ridge(Xf[tri], y[tri], ALPHA), Xf[te])
            q0 = float(np.mean((y[te] - p0) ** 2))
            d = float(np.mean((y[te] - p1) ** 2)) - q0
            deltas.append(d)
            fold_d.append(d)
            sh = int(rng.integers(len(cand) // 4, 3 * len(cand) // 4))
            Xp = np.column_stack([Xbase, np.roll(cand, sh, axis=0)])
            pp = predict_ridge(fit_ridge(Xp[tri], y[tri], ALPHA), Xp[te])
            plac.append(float(np.mean((y[te] - pp) ** 2)) - q0)
        fold_means.append(float(np.mean(fold_d)))
    return {
        "deltas": np.array(deltas),
        "placebo": np.array(plac),
        "fold_means": np.array(fold_means),
    }


def oos_readout(Xbase, cand, y, folds):
    """Full-model OOS corr / HAC cov-t / net-of-1-tick sign Sharpe, pooled over test folds (interpretable
    effect size regardless of the gate verdict)."""
    Xf = (
        np.column_stack([Xbase, cand.reshape(len(cand), -1)])
        if cand is not None
        else Xbase
    )
    P, Y = [], []
    for tr, te in folds:
        P.append(predict_ridge(fit_ridge(Xf[tr], y[tr], ALPHA), Xf[te]))
        Y.append(y[te])
    f, r = np.concatenate(P), np.concatenate(Y)
    pos = np.sign(f)
    pnl = pos * r - np.abs(np.diff(pos, prepend=0.0)) * TICK_COST
    return {
        "oos_corr": float(np.corrcoef(f, r)[0, 1]),
        "cov_t_hac5": cov_t_hac(f, r),
        "net_sign_mean_bp": float(pnl.mean() * 1e4),
        "n": int(len(r)),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    r = df["sumret"].to_numpy(float)
    fab = df["sumret_ind"].to_numpy(float) > 0
    sess = (hour >= 9) & (hour <= 15)

    # forward SESSION-bar target: r[t+1], keep rows where t and t+1 are same-date session, neither fabricated
    r_fwd = np.roll(r, -1)
    same = np.roll(date, -1) == date
    ok = sess & np.roll(sess, -1) & same & ~fab & ~np.roll(fab, -1)
    ok[-1] = False

    mom = mom_features(r)
    flow_cols = [
        "vd_vix",
        "vd_vvix",
        "vd_vix3m",
        "vd_voldemand_spx_open_only",
        "vd_voldemand_all_open_only",
        "vd_voldemand_spx_open_and_close",
        "vd_voldemand_all_open_and_close",
    ]
    flow_cols = [c for c in flow_cols if c in df.columns]
    flow_lv = np.column_stack([zscore_causal(df[c].to_numpy(float)) for c in flow_cols])
    slope = zscore_causal((df["vd_vix3m"] - df["vd_vix"]).to_numpy(float))
    flow_lv = np.column_stack([flow_lv, slope])
    # gamma-regime interaction: last-bar return x standardized vol demand (the sign that sets dealer gamma)
    demand = [
        c
        for c in ("vd_voldemand_spx_open_only", "vd_vvix", "vd_voldemand_all_open_only")
        if c in df.columns
    ]
    m1 = mom[:, 0]
    flow_ix = np.column_stack(
        [m1 * zscore_causal(df[c].to_numpy(float)) for c in demand]
    )

    idx = np.where(ok)[0]
    y = r_fwd[idx]
    Xmom = mom[idx]
    Flv = flow_lv[idx]
    Fix = flow_ix[idx]
    folds = purged_walk_forward(len(idx), n_folds=5, embargo=0.01)

    res: dict = {"n_rows": int(len(idx)), "n_folds": len(folds), "arms": {}}
    res["mom_base_readout"] = oos_readout(Xmom, None, y, folds)
    for name, cand in (
        ("flow_levels", Flv),
        ("flow_x_mom", Fix),
        ("flow_all", np.column_stack([Flv, Fix])),
    ):
        sc = score_dir(Xmom, cand, y, folds)
        passed, comp = significance_gate(sc, k=2.0)
        rd = oos_readout(Xmom, cand, y, folds)
        res["arms"][name] = {"gate_pass": bool(passed), **comp, "readout": rd}
        print(
            f"[{name}] GATE={'PASS' if passed else 'fail'} "
            f"mean_dMSE={comp['mean']:+.3e} ci0={comp['ci_excludes_0']} "
            f"repl={comp['repl_frac']:.2f} beats_plac={comp['beats_placebo']} "
            f"| corr={rd['oos_corr']:+.4f} cov_t={rd['cov_t_hac5']:+.2f} "
            f"net={rd['net_sign_mean_bp']:+.3f}bp",
            flush=True,
        )
    mb = res["mom_base_readout"]
    print(
        f"[mom_base] corr={mb['oos_corr']:+.4f} cov_t={mb['cov_t_hac5']:+.2f} net={mb['net_sign_mean_bp']:+.3f}bp n={mb['n']}",
        flush=True,
    )
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
