"""Invariant checks for the exog transform choice — and why NOT to pick one by OOS score.

This module started life as a five-arm horse race to find "the best universal semantic
transform" by out-of-sample R². **That was the wrong instrument, and the race was never run.**
Recorded here because the reasoning is the reusable part:

* By the time this was written the study had scored **~100+ specifications against the same
  218,934-bar OOS residual** (a 4-point ridge ladder, 8 bucket signals, 13 granularity points x 3
  axes, 13 gain-channel arms, a 6-point sparse ladder, 13 interaction arms, 4 scalings x 3 arms x
  2, 4 product penalties, 5 voldemand variants). There is no unused data left in this panel.
* The candidate transforms differ by ~0.0005-0.003 R2 — *smaller than the selection noise of the
  search that preceded them*. And 219k autocorrelated bars are ~4.5k independent days, so the
  real noise floor is well above the naive one. A max over five correlated arms at that
  separation is not evidence; reporting its per-arm DM-t would be a garden-of-forking-paths
  result.
* A transform is anyway not the kind of thing OOS score should decide. It is a *specification*
  choice that follows from the feature's data-generating mechanism (the Bartlett
  variance-stabilizing transform ``integral d(mu)/sqrt(g(mu))`` for ``Var(x) = g(E x)``): sqrt is
  *derived* for chi-square-like quadratic variation (RV, bipow) and for count-like series
  (numobs, volume); log for multiplicative positives (vix, spreads); asinh — the signed analogue
  of log — for a signed heavy-tailed flow with multiplicative scale (voldemand). The rule table
  is therefore the right *shape* of solution; its defect was that rule 5 was a fall-through
  default rather than a derived transform. There is no single best universal transform because
  the 41 features do not share a distribution family (counts, quadratic variation, signed flows,
  a bounded sentiment index, an index level).

**So what should be universal is the invariance requirements, not the function.** These are
checkable per column, cost no inferential budget (they are not performance comparisons), and are
what ``§8``'s voldemand fix was actually justified by:

1. **Signed-safe** — 7 of 41 raw features are signed and the sign is the signal (``sumret``,
   |IC| 0.108). Rules out "just log everything"; rule 6 already needs ``clip(lower=1e-12)``.
2. **Zero-safe** — defined and smooth at 0 (``voldemand`` is 61% exact zeros).
3. **Scale-equivariant with a causal scale** — ``T(c*x) ~ T(x)`` for ``c > 0``, so a units change
   or secular level drift cannot move the feature. This is the invariant ``voldemand`` violated.
4. **Non-degenerate divisor** — any scale estimated off values that carry dispersion, and
   floored. The whole Part-3 bug class is this one requirement.
5. **Monotone** — so tree models are unaffected.
6. **Tail-magnitude preserving** — the requirement that cuts *against* maximal robustness, and
   why rank-Gauss is not automatically the answer: it maps a 10-sigma and a 3-sigma move to
   nearly the same place, and for a volatility target that may be discarding the signal.
   (Independently, §8.4 measured that rank-Gauss is not even robust downstream: 61% zeros become
   a point mass whose rolling MA has near-zero dispersion, so the degenerate divisor returns at
   the scaler — max |z| 278.)

**If an empirical comparison is wanted anyway**, the protocol is: pre-register the candidates and
the decision rule; hold out an era (all search on <= 2020, one scored decision on 2021-2024 —
this panel has no clean holdout left for this purpose, which is a real cost of how the study
ran); correct for the search with Romano-Wolf step-down or Hansen SPA over the *maximum*
statistic, block-bootstrapped for autocorrelation; and require the invariant checks above to pass
first, since a candidate that fails scale-equivariance is out regardless of score.

What remains here and is legitimate to run: :func:`build_value_block` (rebuild the 246 exog value
columns under an alternative transform) and :func:`_health` / the tail statistics, which are
*invariant diagnostics* rather than a ranking. ``--stage score`` and ``--stage recheck`` still
work, but their all-exog R2 columns must not be used to select a transform — read them only as
"does this candidate keep the panel healthy", which is a pass/fail on requirement 3-4.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, dm_test  # noqa: E402
from analysis.alpha_panel import CACHE_DIR  # noqa: E402
from analysis.wf import r2_oos, walk_forward  # noqa: E402
from src.backtest.executor import _build_scale_guards  # noqa: E402
from src.data.loading import ALL_FEATURES, load_raw_data  # noqa: E402
from src.features.extractors.har import generate_har_features, resolve_har_lags  # noqa: E402
from src.features.transforms.rank_gauss import rolling_rank_gauss  # noqa: E402
from src.features.transforms.scaling import rolling_robust_scale  # noqa: E402
from src.features.transforms.target import (  # noqa: E402
    PERIODS_PER_DAY,
    asinh_stabilize,
    diurnal_adjust,
    robust_transform,
    rolling_winsorize,
)

TRAIN_WINDOW_DAYS = 250
TWP = TRAIN_WINDOW_DAYS * PERIODS_PER_DAY
CANDIDATES = ("rules", "asinh", "asinh_nodiur", "signed_sqrt", "rank_gauss")
BUCKETS = [
    "moments",
    "liquidity",
    "implied_vol",
    "market_vw",
    "market_ew",
    "vol_demand",
    "sentiment",
]
OUT = "results/alpha_manifestation"
FIXED_CACHE = os.path.join(
    CACHE_DIR, "fixed"
)  # the §8-fixed panel is the production baseline


def _scale(x: pd.Series) -> pd.Series:
    """Causal, adaptive, floored scale over active values (see :func:`asinh_stabilize`)."""
    a = x.abs().where((x != 0) & x.notna())
    s = a.ewm(halflife=TWP, min_periods=50, ignore_na=True).mean().shift(1).ffill()
    s = np.maximum(s, 0.01 * s.expanding(min_periods=50).median()).replace(0.0, np.nan)
    med = s.median()
    return s.fillna(med if pd.notna(med) and med > 0 else 1.0)


def build_value_block(candidate: str) -> tuple[np.ndarray, list[str]]:
    """The 246 scaled exog *value* columns under one candidate transform."""
    df = load_raw_data("data", allow_missing=True)
    df[ALL_FEATURES] = df[ALL_FEATURES].ffill()
    df = df.dropna(subset=["RV"]).reset_index(drop=True)
    adj_rv, _ = robust_transform(
        df, "RV", is_target=True, use_diurnal=True, winsor_window=240
    )
    df["adj_RV"] = adj_rv

    cols: list[str] = []
    for col in ALL_FEATURES:
        x = df[col]
        if candidate == "rules":
            a, _ = robust_transform(df, col, use_transform=True, use_diurnal=True)
        elif candidate == "rank_gauss":
            a = pd.Series(
                rolling_rank_gauss(df[[col]].to_numpy(dtype=np.float64), window=TWP)[
                    :, 0
                ],
                index=df.index,
            )
        else:
            if (
                candidate == "asinh"
            ):  # keep the conditional diurnal step, then stabilise
                x, _ = diurnal_adjust(
                    x, df["time_of_day"], bool((x.dropna() < 0).any())
                )
            if candidate == "signed_sqrt":
                s = _scale(x)
                a = rolling_winsorize(np.sign(x) * np.sqrt(np.abs(x / s)), window=240)
            else:
                a = rolling_winsorize(asinh_stabilize(x), window=240)
        df[f"adj_{col}"] = a.fillna(a.median())
        cols.append(f"adj_{col}")
        df[f"{col}_avail"] = df[col].notna().astype("float64")  # for the scale guards

    df, names = generate_har_features(df, target_col="adj_RV", exog_cols=cols)
    names = [n for n in names if not n.startswith("har_ma_")]
    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = df[names].to_numpy(dtype=np.float64)
    ref, fixed = _build_scale_guards(X, df, names)
    X = rolling_robust_scale(X, TWP, ref_iqr=ref, fixed_cols=fixed)
    return X, names


def _panel_for(candidate: str, base) -> np.ndarray:
    """Splice a candidate's value block into the production panel (exact: scaling is per-column)."""
    if candidate == "rules":
        return base.X
    X, names = build_value_block(candidate)
    val = base.cols("value")
    order = [names.index(base.names[j]) for j in val]
    Xp = base.X.copy()
    Xp[:, val] = X[:, order].astype(np.float32)
    return Xp


def _health(Xp: np.ndarray, base, e: np.ndarray) -> float:
    """§7.1 downstream health: unclipped, monthly-refit, center-only base R² (was −0.27)."""
    cols = np.concatenate(
        [
            base.cols("har"),
            np.array([j for j in base.cols("value") if base.window[j] in (1, 25, 625)]),
        ]
    )
    Xb = np.ascontiguousarray(Xp[TW:, cols], dtype=np.float64)
    pred = np.full(len(e) - TW, np.nan)
    for t0 in range(TW, len(e), 21 * PERIODS_PER_DAY):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + 21 * PERIODS_PER_DAY, len(e))
        mu = Xb[tr].mean(0)
        Ztr, Zte = Xb[tr] - mu, Xb[t0:t1] - mu
        g = Ztr.T @ Ztr + 3000.0 * np.eye(Ztr.shape[1])
        b = np.linalg.solve(g, Ztr.T @ (e[tr] - e[tr].mean()))
        pred[t0 - TW : t1 - TW] = Zte @ b + e[tr].mean()
    return r2_oos(e[TW:], pred)


def stage_score() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.environ["ALPHA_PANEL_CACHE"] = FIXED_CACHE
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    base = ap.load_panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    y = e[TW:]
    rows = []
    for cand in CANDIDATES:
        Xp = _panel_for(cand, base)
        a = np.abs(Xp[:, base.cols("value")])
        r = {
            "candidate": cand,
            "max_abs_z": float(a.max()),
            "p99.99": float(np.percentile(a, 99.99)),
            "health_r2": _health(Xp, base, e),
        }
        for b in BUCKETS + ["all"]:
            if b == "all":
                cols = np.concatenate([base.cols("value"), base.cols("indicator")])
            else:
                cols = np.concatenate(
                    [base.cols("value", b), base.cols("indicator", b)]
                )
            s = walk_forward(
                Xp[TW:, cols].astype(np.float64), e, TW, alpha=3000.0, refit_every=48
            )
            r[f"corr_{b}"] = float(np.corrcoef(s, y)[0, 1])
            r[f"r2_{b}"] = r2_oos(y, s)
            if b == "all":
                np.save(os.path.join(CACHE_DIR, f"ut_all_{cand}.npy"), s)
        rows.append(r)
        print(
            f"  {cand:13s} max|z| {r['max_abs_z']:7.1f}  health {r['health_r2']:+.5f}  "
            f"all-exog corr {r['corr_all']:+.4f} R2 {r['r2_all']:+.5f}  "
            f"| vd {r['r2_vol_demand']:+.5f} mom {r['r2_moments']:+.5f} liq {r['r2_liquidity']:+.5f}",
            flush=True,
        )
    d = pd.DataFrame(rows)
    ref = np.load(os.path.join(CACHE_DIR, "ut_all_rules.npy"))
    d["all_dm_t_vs_rules"] = [
        dm_test(y, np.load(os.path.join(CACHE_DIR, f"ut_all_{c}.npy")), ref)
        for c in d["candidate"]
    ]
    print("\nall-exog composite vs the rule table (DM-t, + = better):")
    for _, r in d.iterrows():
        print(
            f"  {r['candidate']:13s} ΔR² {r['r2_all'] - d.r2_all[0]:+.5f}   DM-t {r['all_dm_t_vs_rules']:+6.2f}"
        )
    d.to_csv(f"{OUT}/universal_transform_scores.csv", index=False)
    print(f"wrote {OUT}/universal_transform_scores.csv")


def stage_recheck() -> None:
    """Does §2's central claim — OOS accuracy monotone in feature count — survive the transform?"""
    os.makedirs(OUT, exist_ok=True)
    os.environ["ALPHA_PANEL_CACHE"] = FIXED_CACHE
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    base = ap.load_panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    y = e[TW:]
    val = base.cols("value")
    rows = []
    for cand in CANDIDATES:
        Xp = _panel_for(cand, base)
        X = np.ascontiguousarray(Xp[TW:, val], dtype=np.float64)
        n, pv = len(e), X.shape[1]
        ks = (5, 25, 100, pv)
        preds = {k: np.empty(n - TW) for k in ks}
        Sxx = X[:TW].T @ X[:TW]
        Sxy = X[:TW].T @ e[:TW]
        sx, sy = X[:TW].sum(0), float(e[:TW].sum())
        syy = float(e[:TW] @ e[:TW])
        coef: dict[int, tuple] = {}
        for i in range(n - TW):
            t = TW + i
            if i % 48 == 0:
                g = Sxx - np.outer(sx, sx) / TW
                r = Sxy - sx * (sy / TW)
                ic = r / np.sqrt(
                    np.maximum(np.diag(g), 1e-12) * max(syy - sy * sy / TW, 1e-12)
                )
                rank = np.argsort(-np.abs(np.nan_to_num(ic)))
                for k in ks:
                    sel = np.sort(rank[:k])
                    b = np.linalg.solve(
                        g[np.ix_(sel, sel)] + 3000.0 * np.eye(k), r[sel]
                    )
                    coef[k] = (sel, b, sy / TW - float(sx[sel] @ b) / TW)
            for k in ks:
                sel, b, c0 = coef[k]
                preds[k][i] = float(X[t, sel] @ b) + c0
            xo, xi = X[t - TW], X[t]
            Sxx += np.outer(xi, xi) - np.outer(xo, xo)
            Sxy += xi * e[t] - xo * e[t - TW]
            sx += xi - xo
            sy += float(e[t]) - float(e[t - TW])
            syy += float(e[t]) ** 2 - float(e[t - TW]) ** 2
        r2s = {k: r2_oos(y, preds[k]) for k in ks}
        mono = all(r2s[ks[i]] <= r2s[ks[i + 1]] for i in range(len(ks) - 1))
        print(
            f"  {cand:13s} R2 k=5 {r2s[5]:+.5f}  k=25 {r2s[25]:+.5f}  k=100 {r2s[100]:+.5f}  "
            f"k={pv} {r2s[pv]:+.5f}  | monotone in k: {mono}  | DM-t dense-vs-top5 "
            f"{dm_test(y, preds[pv], preds[5]):+5.2f}",
            flush=True,
        )
        rows.append(
            {
                "candidate": cand,
                **{f"r2_k{k}": r2s[k] for k in ks},
                "monotone": mono,
                "dm_t_dense_vs_top5": dm_test(y, preds[pv], preds[5]),
            }
        )
    pd.DataFrame(rows).to_csv(f"{OUT}/universal_transform_recheck.csv", index=False)
    print(f"wrote {OUT}/universal_transform_recheck.csv")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["score", "recheck"], required=True)
    a = ap_.parse_args()
    {"score": stage_score, "recheck": stage_recheck}[a.stage]()
