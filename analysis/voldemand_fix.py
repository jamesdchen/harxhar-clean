"""Fix `voldemand`'s non-stationary scale at the feature level, and prove it robustly.

**The defect.** `voldemand_*` is a signed, zero-inflated, sparsely-observed flow whose scale
grows ~11x across the sample (`voldemand_spx_open_only` IQR 51k in 2011 -> 572k in 2023,
median 0 -> 472k; the 0DTE build-out). Three pipeline steps then fail in sequence:

1. :func:`~src.features.transforms.target.diurnal_adjust` divides a signed feature by its
   per-slot rolling **std**. For an ffilled, 60%-zero series that std is degenerate most of the
   time, so ``DIURNAL_STD_FLOOR_FRAC`` pins the divisor to its floor on **73.8% of rows** — the
   "diurnal adjustment" degenerates into division by one global constant, which cannot track a
   secular scale expansion.
2. :func:`~src.features.transforms.target.apply_semantic_transform` rule 5 returns **identity**
   for signed features, so there is no variance stabilizer at all. (Flagged in the Part-3
   session notes as "the deeper mis-design; asinh would fit, but the hurdle sufficed".)
3. :func:`~src.features.transforms.target.rolling_winsorize` cannot bound it either: a trailing
   5-95% band over 240 bars tracks a *trending* series, so it clips only 6.9% of rows. Max
   |adj| stays at **487** for 2021-23 while 2011-16 falls to 6.2.

Downstream, ``_build_scale_guards`` floors the *divisor* of the rolling-robust scaler but
cannot bound a *numerator* that is already 487 sigma, so the panel carries |z| up to 2011 —
0.04% of cells, all voldemand, 83% of them in 2023 — which is enough to break any fit that
does not refit every bar (it distorted every arm of the interaction study, §7.1).

**The candidate fixes** (all strictly causal; ``s_t`` uses only data before ``t``):

* ``status_quo``    — today's chain, for reference.
* ``asinh_diurnal`` — ``asinh(x / s_t)`` first, *then* the existing diurnal + winsorize. The
  stabilized series is O(1), so the per-slot std is no longer degenerate and the diurnal step
  becomes meaningful again instead of a constant divide.
* ``asinh_only``    — ``asinh(x / s_t)`` + winsorize, no diurnal at all.
* ``rank_gauss``    — :func:`~src.features.transforms.rank_gauss.rolling_rank_gauss` on the raw
  series. Division-free and invariant to any monotone level/scale drift *by construction*, so
  the failure mode cannot recur. The repo built this and never wired it in.
* ``asinh_fast``    — same as ``asinh_only`` but with a 60-day scale halflife instead of 250,
  to test how fast the denominator should chase the trend.

``s_t`` is an EWMA of ``|x|`` over **active** (non-zero, available) observations, halflife 250
days, shifted one bar, floored at 1% of its own expanding median — it adapts to the 11x growth
and cannot go degenerate.

**The evaluation** — a fix has to clear all four bars, not one:

1. **Tails.** max |z| and p99.9 by era in the finished panel.
2. **Downstream health.** Re-run the §7.1 ``center`` specification (no clipping, monthly refit)
   *with* voldemand included. Status quo poisons it (base R² −0.293 vs +0.019 without
   voldemand); a real fix leaves it healthy with the columns in.
3. **Forecast value.** The `vol_demand` bucket's own walk-forward OOS corr / R² on the HAR
   residual (status quo: +0.0390 / −0.00026), and the all-exog composite (+0.1897 / +0.03471).
4. **No collateral damage.** Only the 4 voldemand raw columns are touched, so every other
   bucket must be bit-identical; asserted, not assumed.

Splicing rather than rebuilding: ``rolling_robust_scale`` and ``_build_scale_guards`` are both
per-column, so rebuilding only the voldemand columns and substituting them into the cached
panel is exact — and turns an 8-minute rebuild per variant into ~20 seconds.

Usage
-----
    python analysis/voldemand_fix.py --stage diagnose   # per-stage tails + why it breaks
    python analysis/voldemand_fix.py --stage evaluate   # all variants against the four bars
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, dm_test  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.wf import r2_oos, walk_forward  # noqa: E402
from src.backtest.executor import ZERO_INFLATED_FRAC, _build_scale_guards  # noqa: E402
from src.data.loading import SUBGROUPS, load_raw_data  # noqa: E402
from src.features.extractors.har import generate_har_features, resolve_har_lags  # noqa: E402
from src.features.transforms.rank_gauss import rolling_rank_gauss  # noqa: E402
from src.features.transforms.scaling import rolling_robust_scale  # noqa: E402
from src.features.transforms.target import (  # noqa: E402
    PERIODS_PER_DAY,
    robust_transform,
    rolling_winsorize,
)

VD = SUBGROUPS["vol_demand"]
TRAIN_WINDOW_DAYS = 250
VARIANTS = ("status_quo", "asinh_diurnal", "asinh_only", "asinh_fast", "rank_gauss")
OUT = "results/alpha_manifestation"


def _active_scale(x: pd.Series, halflife_days: int = 250) -> pd.Series:
    """Causal adaptive scale: EWMA of |x| over ACTIVE obs, shifted, floored.

    Active = observed and non-zero. The zero point-mass carries no scale information (the same
    reasoning as ``_build_scale_guards``'s zero-inflation branch), and the EWMA tracks a secular
    expansion that any fixed constant cannot. Floored at 1% of its own expanding median so it
    can never become a degenerate divisor.
    """
    a = x.abs().where((x != 0) & x.notna())
    s = a.ewm(halflife=halflife_days * PERIODS_PER_DAY, min_periods=50, ignore_na=True).mean()
    s = s.shift(1).ffill()
    floor = 0.01 * s.expanding(min_periods=50).median()
    s = np.maximum(s, floor).replace(0.0, np.nan)
    med = s.median()
    return s.fillna(med if pd.notna(med) and med > 0 else 1.0)


def build_adj(df: pd.DataFrame, col: str, variant: str) -> pd.Series:
    """The adjusted (pre-MA) feature for one voldemand column under one variant."""
    if variant == "status_quo":
        # ``signed_stabilizer=False`` is load-bearing: the asinh gate this module motivated is now
        # the production DEFAULT in ``robust_transform``, so without disabling it "status_quo"
        # would silently return the *fixed* series and the pre-fix baseline would vanish — which
        # is exactly what the diagnostics back-test caught (I1 and I2 collapsed onto one arm).
        adj, _ = robust_transform(
            df, col, use_transform=True, use_diurnal=True, signed_stabilizer=False
        )
        return adj
    if variant == "rank_gauss":
        z = rolling_rank_gauss(
            df[[col]].to_numpy(dtype=np.float64),
            window=TRAIN_WINDOW_DAYS * PERIODS_PER_DAY,
        )[:, 0]
        return pd.Series(z, index=df.index)

    x = df[col]
    # x / s_t is the ratio a growing flow series wants; asinh then bounds the tails (log-like
    # far out, linear near 0, and asinh(0) = 0 so the zero mass is untouched).
    s = _active_scale(x, halflife_days=60 if variant == "asinh_fast" else 250)
    z = pd.Series(np.arcsinh(x / s), index=df.index)
    if variant == "asinh_diurnal":
        tmp = df[[col, "time_of_day"]].copy()
        tmp[col] = z
        adj, _ = robust_transform(
            tmp, col, use_transform=False, use_diurnal=True, winsor_window=240
        )
        return adj
    return rolling_winsorize(z, window=240)


def build_vd_block(variant: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(scaled_columns, names, timestamps) for the voldemand block under one variant.

    Mirrors ``alpha_panel.build_panel`` exactly for these columns — impute-and-indicate, hurdle
    indicators, the 6 HAR windows, the scale guards and the rolling-robust prescale — so the
    result is drop-in substitutable into the cached panel.
    """
    df = load_raw_data("data", allow_missing=True)
    df[VD] = df[VD].ffill()
    df = df.dropna(subset=["RV"]).reset_index(drop=True)
    adj_rv, _ = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj_rv

    cols: list[str] = []
    for col in VD:
        a = build_adj(df, col, variant)
        df[f"adj_{col}"] = a.fillna(a.median())
        cols.append(f"adj_{col}")
    for col in VD:
        df[f"{col}_avail"] = df[col].notna().astype("float64")
        cols.append(f"{col}_avail")
    for col in VD:
        v = df[col].to_numpy(dtype=np.float64)
        fin = v[~np.isnan(v)]
        if fin.size and float(np.mean(fin == 0.0)) >= ZERO_INFLATED_FRAC:
            df[f"{col}_active"] = (np.nan_to_num(v) != 0.0).astype("float64")
            cols.append(f"{col}_active")

    df, names = generate_har_features(df, target_col="adj_RV", exog_cols=cols)
    names = [n for n in names if not n.startswith("har_ma_")]
    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = df[names].to_numpy(dtype=np.float64)
    ref, fixed = _build_scale_guards(X, df, names)
    X = rolling_robust_scale(
        X, TRAIN_WINDOW_DAYS * PERIODS_PER_DAY, ref_iqr=ref, fixed_cols=fixed
    )
    return X, names, df["t"].to_numpy()


def stage_diagnose() -> None:
    """Per-stage tails for the worst column, and the two mechanisms behind them."""
    df = load_raw_data("data", allow_missing=True)
    df[VD] = df[VD].ffill()
    df = df.dropna(subset=["RV"]).reset_index(drop=True)
    yr = df["t"].dt.year
    rows = []
    for col in VD:
        for variant in VARIANTS:
            a = build_adj(df, col, variant).abs()
            rows.append(
                {
                    "column": col,
                    "variant": variant,
                    "p99.9": float(a.quantile(0.999)),
                    "max": float(a.max()),
                    "max_2011_2016": float(a[(yr >= 2011) & (yr <= 2016)].max()),
                    "max_2021_2023": float(a[yr >= 2021].max()),
                }
            )
        print(f"  {col}", flush=True)
    d = pd.DataFrame(rows)
    print("\nAdjusted-feature tails (pre-MA, pre-scaler):")
    print(
        d.pivot_table(index="variant", values=["p99.9", "max", "max_2011_2016", "max_2021_2023"])
        .round(2)
        .to_string()
    )
    d.to_csv(f"{OUT}/voldemand_stage_tails.csv", index=False)
    print(f"\nwrote {OUT}/voldemand_stage_tails.csv")


def stage_evaluate() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_full = np.load(_resid_path())["e"]
    vd_cols = np.concatenate([p.cols("value", "vol_demand"), p.cols("indicator", "vol_demand")])
    other = np.array([j for j in range(len(p.names)) if j not in set(vd_cols.tolist())])
    all_exog = np.concatenate([p.cols("value"), p.cols("indicator")])
    rows = []
    for variant in VARIANTS:
        X, names, t = build_vd_block(variant)
        assert len(X) == len(p.X), (len(X), len(p.X))
        order = [names.index(p.names[j]) for j in vd_cols]  # align to panel column order
        if variant == "status_quo":
            # validates the whole splice machinery: rebuilding these columns standalone must
            # reproduce the cached panel exactly, or nothing below is comparable
            assert np.allclose(
                X[:, order].astype(np.float32), p.X[:, vd_cols], atol=1e-4, rtol=1e-3
            ), "status_quo splice does not reproduce the cached panel"
            print("    splice check: status_quo reproduces the cached panel", flush=True)
        Xp = p.X.copy()
        Xp[:, vd_cols] = X[:, order].astype(np.float32)
        assert np.array_equal(Xp[:, other], p.X[:, other]), "collateral damage outside voldemand"

        a = np.abs(Xp[:, vd_cols])
        tails = (float(a.max()), float(np.percentile(a, 99.99)))

        # (3) forecast value: the bucket alone, and the all-exog composite
        sig_vd = walk_forward(
            Xp[TW:, vd_cols].astype(np.float64), e_full, TW, alpha=3000.0, refit_every=48
        )
        sig_all = walk_forward(
            Xp[TW:, all_exog].astype(np.float64), e_full, TW, alpha=3000.0, refit_every=48
        )
        y = e_full[TW:]

        # (2) downstream health: the unclipped `center` spec WITH voldemand in the base
        base_cols = np.concatenate(
            [p.cols("har"), np.array([j for j in p.cols("value") if p.window[j] in (1, 25, 625)])]
        )
        Xb = np.ascontiguousarray(Xp[TW:, base_cols], dtype=np.float64)
        pred = np.full(len(e_full) - TW, np.nan)
        for t0 in range(TW, len(e_full), 21 * PERIODS_PER_DAY):
            tr = slice(t0 - TW, t0)
            t1 = min(t0 + 21 * PERIODS_PER_DAY, len(e_full))
            mu = Xb[tr].mean(0)
            Ztr, Zte = Xb[tr] - mu, Xb[t0:t1] - mu
            g = Ztr.T @ Ztr + 3000.0 * np.eye(Ztr.shape[1])
            b = np.linalg.solve(g, Ztr.T @ (e_full[tr] - e_full[tr].mean()))
            pred[t0 - TW : t1 - TW] = Zte @ b + e_full[tr].mean()

        r = {
            "variant": variant,
            "panel_max_abs_z": tails[0],
            "panel_p99.99": tails[1],
            "vd_bucket_corr": float(np.corrcoef(sig_vd, y)[0, 1]),
            "vd_bucket_r2": r2_oos(y, sig_vd),
            "all_exog_corr": float(np.corrcoef(sig_all, y)[0, 1]),
            "all_exog_r2": r2_oos(y, sig_all),
            "unclipped_base_r2": r2_oos(y, pred),
        }
        rows.append(r)
        print(
            f"  {variant:14s} max|z| {r['panel_max_abs_z']:8.1f} | vd bucket corr "
            f"{r['vd_bucket_corr']:+.4f} R2 {r['vd_bucket_r2']:+.5f} | all-exog corr "
            f"{r['all_exog_corr']:+.4f} R2 {r['all_exog_r2']:+.5f} | unclipped base R2 "
            f"{r['unclipped_base_r2']:+.5f}",
            flush=True,
        )
        np.save(os.path.join(_cache_dir(), f"vd_sig_all_{variant}.npy"), sig_all)

    d = pd.DataFrame(rows)
    sq = np.load(os.path.join(_cache_dir(), "vd_sig_all_status_quo.npy"))
    y = e_full[TW:]
    d["all_exog_dm_t_vs_status_quo"] = [
        dm_test(y, np.load(os.path.join(_cache_dir(), f"vd_sig_all_{v}.npy")), sq)
        for v in d["variant"]
    ]
    print("\nall-exog composite, DM-t vs status quo (+ = better):")
    for _, r in d.iterrows():
        print(f"  {r['variant']:14s} {r['all_exog_dm_t_vs_status_quo']:+6.2f}")
    d.to_csv(f"{OUT}/voldemand_variants.csv", index=False)
    print(f"wrote {OUT}/voldemand_variants.csv")


def stage_control() -> None:
    """Full control: rebuild the panel WITH the production fix and re-score every bucket.

    The old signals (``bucket_signals.npz``) were built at the same spec (rolling 250-day, daily
    refit, alpha 3000) on the pre-fix panel, and the HAR residual is exog-free so the target is
    identical — the only difference between the two runs is the 4 voldemand columns. Anything
    that moves outside ``vol_demand`` would be a bug, and is asserted against, not hoped for.
    """
    os.makedirs(OUT, exist_ok=True)
    old = load_panel()
    orig_cache = _cache_dir()  # har_resid / bucket_signals live here, not in the fixed cache
    fixed_cache = os.path.join(orig_cache, "fixed")
    os.makedirs(fixed_cache, exist_ok=True)
    os.environ["ALPHA_PANEL_CACHE"] = fixed_cache
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    new = ap.load_panel()
    assert new.names == old.names, "column set changed"

    vd = set(np.concatenate([old.cols("value", "vol_demand"), old.cols("indicator", "vol_demand")]).tolist())
    other = np.array([j for j in range(len(old.names)) if j not in vd])
    n_diff = int((~np.isclose(new.X[:, other], old.X[:, other], atol=1e-5, rtol=1e-4)).sum())
    print(f"  non-voldemand cells differing: {n_diff} (must be 0)", flush=True)
    assert n_diff == 0, "collateral damage outside voldemand"
    print(f"  panel max |z|: {np.abs(old.X).max():.1f} -> {np.abs(new.X).max():.1f}", flush=True)

    e = np.load(os.path.join(orig_cache, "har_resid.npz"))["e"]
    y = e[TW:]
    old_sig = dict(np.load(os.path.join(orig_cache, "bucket_signals.npz")))
    rows = []
    for b in list(old_sig.keys()):
        if b == "all":
            cols = np.concatenate([new.cols("value"), new.cols("indicator")])
        else:
            cols = np.concatenate([new.cols("value", b), new.cols("indicator", b)])
        s_new = walk_forward(
            new.X[TW:, cols].astype(np.float64), e, TW, alpha=3000.0, refit_every=48
        )
        s_old = old_sig[b]
        r = {
            "bucket": b,
            "corr_old": float(np.corrcoef(s_old, y)[0, 1]),
            "corr_new": float(np.corrcoef(s_new, y)[0, 1]),
            "r2_old": r2_oos(y, s_old),
            "r2_new": r2_oos(y, s_new),
            "dm_t_new_vs_old": dm_test(y, s_new, s_old),
        }
        rows.append(r)
        print(
            f"  {b:12s} corr {r['corr_old']:+.4f} -> {r['corr_new']:+.4f}  "
            f"R2 {r['r2_old']:+.5f} -> {r['r2_new']:+.5f}  DM-t {r['dm_t_new_vs_old']:+5.2f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(f"{OUT}/voldemand_fix_control.csv", index=False)
    print(f"wrote {OUT}/voldemand_fix_control.csv")


def _cache_dir() -> str:
    from analysis.alpha_panel import CACHE_DIR

    return CACHE_DIR


def _resid_path() -> str:
    return os.path.join(_cache_dir(), "har_resid.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["diagnose", "evaluate", "control"], required=True)
    a = ap.parse_args()
    {"diagnose": stage_diagnose, "evaluate": stage_evaluate, "control": stage_control}[a.stage]()
