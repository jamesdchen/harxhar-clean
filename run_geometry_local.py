"""Local multi-arm geometry screener — ONE prep, many arms.

specs/spectral_knn_var.py pays the full data prep (load + transforms +
whole-series rolling robust scale, ~30 min locally for all_features) on EVERY
invocation, then backtests a ~2k-bar chunk in minutes. This driver splits
that: the prepared (X, y, baselines, names) matrix is built ONCE via the
executor's own functions (CALLED, never re-derived — the byte-identical
invariants of the spec) and cached to disk; each arm then reuses it
in-process. The chunk, the identity anchor, the incumbent and the metrics
mirror the spec exactly, so a row here is the same experiment as a spec row
— minus the redundant prep.

Usage:
    python run_geometry_local.py euclid:euclid:12 pca960d8:pca:960:25:8 [...]
arm syntax: tag:geometry:view_w[:neighbor_k[:d[:resid]]]
  geometry euclid|har|whiten|tica|pca ; d = embedding/PCA dim (0 = passthrough)
  resid identity (default) | ridge1 (Ridge alpha=1 anchor) | ols0 (masked OLS)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from src.backtest.executor import (
    _build_har_and_calendar,
    _build_scale_guards,
    load_and_transform,
)
from src.data.loading import get_bucket
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.features.extractors.har import resolve_har_lags
from src.features.transforms.scaling import rolling_robust_scale
from src.features.transforms.target import PERIODS_PER_DAY, apply_horizon_shift
from src.models.ridge import fit_predict_ridge
from src.models.spectral_knn import fit_predict_spectral_knn

BUCKET = "all_features"
HORIZON = 1
TRAIN_WIN = 500 * PERIODS_PER_DAY   # 24000 bars, the spec's TRAIN_WIN
START, END, HALO = 24000, 26189, 24000  # first campaign-grid tile
CACHE_DIR = "results"
# HAR_BASE: ladder base for the ARM bucket's HAR features (env; default the
# production base-5). The har_base_sweep battery crowned base-2 — HAR_BASE=2
# rebuilds the state on [1,2,4,...,2048]. The INCUMBENT stays base-5 always:
# it is the audited comparator and must not drift with the arm axis.
HAR_BASE = float(os.environ.get("HAR_BASE", "5"))
# HAR_KERNEL: "mean" (flat rolling windows, the production kernel) or "ewm"
# (EWMA ladder — rungs are HALF-LIVES; smooth forgetting instead of the flat
# window's abrupt one-bar shock exit). Same *_ma_<r> naming, so cap/bands/
# geometry knobs apply unchanged. The incumbent always stays "mean".
HAR_KERNEL = os.environ.get("HAR_KERNEL", "mean")


def _build_ewm_and_calendar(df, exog_cols, half_lives):
    """EWMA mirror of the executor's _build_har_and_calendar.

    Identical naming (target ``har_ma_<h>``, exog ``<col>_ma_<h>``, session
    interactions ``har_ma_<h>_x_{open,close}``) and the same shift(1)
    causality — only the kernel differs: exponentially-weighted mean at
    half-life <h> bars instead of a flat <h>-bar window. Smooth forgetting:
    a shock decays out gradually instead of exiting the window in one bar.
    """
    from src.features.extractors.calendar import add_calendar_features

    feats: dict = {}
    for col in ["adj_RV"] + list(exog_cols):
        prefix = "har" if col == "adj_RV" else col
        s = df[col]
        for h in half_lives:
            feats[f"{prefix}_ma_{h}"] = s.ewm(halflife=float(h)).mean().shift(1)
    df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    names = list(feats.keys()) + add_calendar_features(df)
    for h in half_lives:
        base_col = f"har_ma_{h}"
        for gate, sfx in (("is_open", "open"), ("is_close", "close")):
            nm = f"{base_col}_x_{sfx}"
            df[nm] = df[base_col] * df[gate]
            names.append(nm)
    return df, names


def prepare_full(bucket: str, har_base: int | None = None):
    """Full prepared series (X, y, baselines, names) under the spec invariants.

    Cached uncompressed per bucket — the cache IS the speedup. Everything here
    is the executor's own prep path (load_and_transform,
    _build_har_and_calendar, burn-in drop, horizon shift, scale guards,
    rolling_robust_scale) in _backtest_and_save's exact order. ``har_base``
    != 5 rebuilds all HAR features on that ladder (rungs = powers <= 3125,
    the fixed burn-in) under a distinct cache key; the default (None -> the
    module HAR_BASE for the arm bucket, 5 for `baseline`) keeps the
    production ladder byte-identical.
    """
    base = HAR_BASE if har_base is None else float(har_base)
    kernel = HAR_KERNEL
    if bucket == "baseline":
        base = 5.0   # the audited incumbent's basis — never drifts with the arm axis
        kernel = "mean"
    suffix = "" if base == 5.0 else f"_b{base:g}"
    if kernel != "mean":
        suffix += f"_{kernel}"
    # DIURNAL_MODE: exog per-slot adjustment family — "divide" (production;
    # slot-level for positives, slot-std for signed) or "rank" (per-slot
    # rank-Gauss, the nonparametric marginal endpoint). Target stays divide
    # always (its baseline is the QLIKE reconstruction scale). Distinct
    # cache key; the incumbent (baseline bucket) never drifts.
    diurnal = os.environ.get("DIURNAL_MODE", "divide")
    if bucket == "baseline":
        diurnal = "divide"
    if diurnal != "divide":
        suffix += f"_{diurnal}"
    # PREP_ROWS: build features on only the first N raw rows. Every transform
    # in the prep path is causal (the chunk-fungibility guarantee: row t
    # depends only on rows <= t), so the trimmed values are IDENTICAL to the
    # whole-series build on those rows — at a fraction of the memory. Needed
    # for wide ladders (base-1.5 = ~1800 cols OOM'd the full-series build).
    prep_rows = int(os.environ.get("PREP_ROWS", "0"))
    if prep_rows and bucket != "baseline":
        suffix += f"_r{prep_rows}"
    cache = os.path.join(CACHE_DIR, f"prep_cache_{bucket}{suffix}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["baselines"], list(z["names"])
    t0 = time.time()
    df, adj_exog = load_and_transform(
        "data",
        get_bucket(bucket),
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode=diurnal,
    )
    lags_env = os.environ.get("HAR_LAGS", "")
    if lags_env and bucket != "baseline":
        # explicit rung list (e.g. base-2 UNION daily multiples — the
        # clock-resolution experiment); overrides the base computation
        har_lags = sorted({int(v) for v in lags_env.split(",")})
        suffix += f"_lags{len(har_lags)}"
        cache = os.path.join(CACHE_DIR, f"prep_cache_{bucket}{suffix}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            return z["X"], z["y"], z["baselines"], list(z["names"])
    elif base == 5.0:
        har_lags = None
    else:
        # geometric ladder at any real base > 1: rounded to ints, deduped,
        # capped at the fixed 3125-bar burn-in (arm-invariant index space)
        rungs, v = [], 1.0
        while round(v) <= resolve_har_lags()[-1]:
            rungs.append(int(round(v)))
            v *= base
        har_lags = sorted(set(rungs))
    if prep_rows and bucket != "baseline":
        df = df.iloc[:prep_rows].reset_index(drop=True)
    if kernel == "ewm":
        df, names = _build_ewm_and_calendar(
            df, adj_exog, har_lags or resolve_har_lags()
        )
    else:
        df, names = _build_har_and_calendar(
            df, adj_exog, add_calendar=True, har_lags=har_lags
        )
    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = df[names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)
    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, HORIZON)
    ref_iqr, fixed_cols = _build_scale_guards(X, df, names)
    X = rolling_robust_scale(X, TRAIN_WIN, ref_iqr=ref_iqr, fixed_cols=fixed_cols)
    # Re-scale the imputed exogenous columns using observed rows only. See
    # executor.apply_masked_scaling: masking the ESTIMATE is what the three
    # earlier value-side repairs were all failing to do.
    if os.environ.get("MASKED_SCALE", "1") != "0":
        from src.backtest.executor import apply_masked_scaling
        X = apply_masked_scaling(X, df, names, TRAIN_WIN)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, X=X, y=y, baselines=baselines,
             names=np.array(names, dtype=object))
    print(f"[prep] {bucket}: {X.shape} in {time.time() - t0:.0f}s -> {cache}")
    return X, y, baselines, list(names)


def apply_bands(X, names):
    """Band-pass reparam of every HAR ladder: levels -> scale increments.

    For each series' rung run [ma_r1, ma_r2, ...] (ascending), replace columns
    with [ma_r1, ma_r2 - ma_r1, ma_r3 - ma_r2, ...] — a Haar-style
    orthogonalization of the memory axis. Same span, but nested-MA
    collinearity (which a METRIC cannot zero out, unlike a regression)
    stops double-counting the level factor in every distance. Applies to
    adj_*/avail/active ladders and the target's har_ma_* alike.
    """
    import re

    Xb = X.copy()
    groups: dict = {}
    for j, nm in enumerate(names):
        m = re.match(r"^(.*_ma_|har_ma_)(\d+)$", nm)
        if m and "_x_" not in nm:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), j))
    n_banded = 0
    for run in groups.values():
        run.sort()
        for (_, j_prev), (_, j) in zip(run, run[1:]):
            Xb[:, j] = X[:, j] - X[:, j_prev]
            n_banded += 1
    print(f"[bands] {n_banded} columns banded across {len(groups)} ladders")
    return Xb


def apply_scale_poly(X, names, cap=256, deg=2):
    """OPPOSITE of banding: smooth each ladder along the SCALE axis.

    Each series' rung run <= cap is treated as a memory curve over log(h) and
    projected onto Legendre polynomials of degree <= deg — level / slope /
    curvature of the memory term structure (Nelson-Siegel for memory; the
    rough-vol power-law prior says the curve is near-linear in log-scale).
    Span-REDUCING (deg+1 << n_rungs), so unlike bands this is a genuine
    smoothness-prior compression the downstream PLS cannot undo. Rungs > cap
    are dropped (the minimality verdict). Non-ladder columns pass through.
    Returns (X2, names2).
    """
    import re

    groups: dict = {}
    ladder_cols = set()
    for j, nm in enumerate(names):
        m = re.match(r"^(.*_ma_|har_ma_)(\d+)$", nm)
        if m and "_x_" not in nm:
            ladder_cols.add(j)
            if int(m.group(2)) <= cap:
                groups.setdefault(m.group(1), []).append((int(m.group(2)), j))
    keep = [j for j in range(len(names)) if j not in ladder_cols]
    cols, new_names = [X[:, keep]], [names[j] for j in keep]
    for prefix, run in sorted(groups.items()):
        run.sort()
        if len(run) < 2:
            j = run[0][1]
            cols.append(X[:, [j]])
            new_names.append(names[j])
            continue
        rungs = np.array([r for r, _ in run], dtype=np.float64)
        idx = [j for _, j in run]
        x = np.log(rungs)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        P = np.polynomial.legendre.legvander(x, min(deg, len(run) - 1))
        C = X[:, idx] @ np.linalg.pinv(P).T
        cols.append(C)
        new_names += [f"{prefix}pol{k}" for k in range(C.shape[1])]
    X2 = np.hstack(cols)
    print(f"[scale-poly] {len(names)} -> {X2.shape[1]} cols "
          f"({len(groups)} ladders x deg<={deg})")
    return np.ascontiguousarray(X2), new_names


def apply_shock(X, names, window=256):
    """Append SHOCK-AGE features: bars since each series' trailing-window max.

    The pol1 result says the full ladder's residual value over (level, tilt)
    is shock TIMING — where in the memory curve recent events sit. This
    encodes it directly: for each bar-level series (*_ma_1 columns, already
    shift(1)-causal), the age in bars of the trailing ``window``-bar argmax,
    and the market-wide average age (shocks are largely COMMON — the measured
    factor structure — so the cross-sectional mean is the clean shock clock).
    Threshold-free: argmax needs no cutoff. Ages are in bars; downstream
    standardization handles scale. Returns (X2, names2).
    """
    import re

    from numpy.lib.stride_tricks import sliding_window_view

    idx = [j for j, nm in enumerate(names)
           if re.match(r"^adj_.+_ma_1$", nm)]
    n = X.shape[0]
    ages = []
    for j in idx:
        x = X[:, j]
        age = np.full(n, float(window))
        if n > window:
            sw = sliding_window_view(x, window)      # sw[t] = x[t : t+window]
            # window ending at bar t-1 (inclusive) covers rows [t-window, t)
            am = sw[:-1].argmax(axis=1)              # for t = window .. n-1
            age[window:] = (window - 1) - am
        ages.append(age)
    A = np.stack(ages, axis=1)
    market = A.mean(axis=1, keepdims=True)
    X2 = np.hstack([X, A, market])
    names2 = (list(names)
              + [names[j].replace("_ma_1", "_shockage") for j in idx]
              + ["market_shockage"])
    print(f"[shock] appended {A.shape[1]} series shock-ages + market clock")
    return np.ascontiguousarray(X2), names2


def apply_target_pol(X, names, cap=256, quad=False):
    """Append the TARGET's memory-curve state: (rv_level, rv_tilt) [+ products].

    Level and slope of the target's own base-2 ladder (har_ma_r, r <= cap) in
    log-scale — the explicit roughness/mean-reversion state of the target,
    which the view otherwise carries only via the 12-bar path and
    cross-sectional proxies. ``quad`` appends level^2, tilt^2, level*tilt —
    the roughness-state nonlinearity no linear span of the ladder contains.
    """
    import re

    run = sorted(
        (int(m.group(1)), j)
        for j, nm in enumerate(names)
        if (m := re.match(r"^har_ma_(\d+)$", nm)) and int(m.group(1)) <= cap
    )
    rungs = np.array([r for r, _ in run], dtype=np.float64)
    idx = [j for _, j in run]
    x = np.log(rungs)
    x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    P = np.polynomial.legendre.legvander(x, 1)
    C = X[:, idx] @ np.linalg.pinv(P).T          # (n, 2): level, tilt
    cols = [C]
    new = ["rv_level", "rv_tilt"]
    if quad:
        cols.append(np.stack([C[:, 0] ** 2, C[:, 1] ** 2, C[:, 0] * C[:, 1]], axis=1))
        new += ["rv_level_sq", "rv_tilt_sq", "rv_level_x_tilt"]
    print(f"[target-pol] appended {new}")
    return np.ascontiguousarray(np.hstack([X] + cols)), list(names) + new


def apply_maxladder(X, names, windows=(8, 32, 128)):
    """Append EXTREME-VALUE memory: rolling maxima of target + market series.

    Rolling max at log-spaced windows is nonlinear in the data (not spanned
    by any mean ladder) and encodes the joint size-and-age profile of recent
    shocks — the richer object the single argmax age failed to capture. Kept
    small and legible: the target's bar series (har_ma_1) and the
    cross-sectional mean of the exog bar series (the market shock record).
    """
    import re

    from numpy.lib.stride_tricks import sliding_window_view

    tgt = X[:, names.index("har_ma_1")]
    ex1 = [j for j, nm in enumerate(names) if re.match(r"^adj_.+_ma_1$", nm)]
    mkt = X[:, ex1].mean(axis=1)
    cols, new = [], []
    for label, s in (("rv", tgt), ("market", mkt)):
        for w in windows:
            m = np.full(len(s), np.nan)
            sw = sliding_window_view(s, w).max(axis=1)
            m[w - 1:] = sw
            m[: w - 1] = s[: w - 1]  # warm-up: expanding max is fine causally
            np.maximum.accumulate(s[:w - 1], out=m[: w - 1])
            cols.append(m)
            new.append(f"{label}_max_{w}")
    print(f"[max-ladder] appended {new}")
    return (np.ascontiguousarray(np.hstack([X, np.stack(cols, axis=1)])),
            list(names) + new)


def apply_volofvol(X, names, windows=(8, 32, 128)):
    """Append DISPERSION memory: rolling std of target + market bar series.

    Mean ladders store each window's level; this stores its burstiness —
    vol-of-vol at log windows, quadratic in the data so not spanned by any
    mean ladder. Causal: the bar series is already shift(1).
    """
    import re

    from numpy.lib.stride_tricks import sliding_window_view

    tgt = X[:, names.index("har_ma_1")]
    ex1 = [j for j, nm in enumerate(names) if re.match(r"^adj_.+_ma_1$", nm)]
    mkt = X[:, ex1].mean(axis=1)
    cols, new = [], []
    for label, s in (("rv", tgt), ("market", mkt)):
        for w in windows:
            v = np.zeros(len(s))
            v[w - 1:] = sliding_window_view(s, w).std(axis=1)
            cols.append(v)
            new.append(f"{label}_vv_{w}")
    print(f"[vol-of-vol] appended {new}")
    return (np.ascontiguousarray(np.hstack([X, np.stack(cols, axis=1)])),
            list(names) + new)


def apply_session_memory(X, names):
    """Append CALENDAR-ALIGNED memory: today-so-far / yesterday / this-week.

    Rolling bar-count windows straddle day boundaries incoherently; the
    repo's strongest regime finding is CLOCK-anchored. For target + market:
    the expanding mean since session start (today-so-far), the completed
    prior day's mean (yesterday), and the trailing 5 completed days' mean
    (week). Day boundaries from the calendar `hour` column's wrap. Causal:
    built from the shift(1) bar series; 'yesterday'/'week' use only
    completed days.
    """
    import re

    hour = X[:, names.index("hour")]
    day_id = np.cumsum(np.r_[1, (np.diff(hour) < 0).astype(int)])
    tgt = X[:, names.index("har_ma_1")]
    ex1 = [j for j, nm in enumerate(names) if re.match(r"^adj_.+_ma_1$", nm)]
    mkt = X[:, ex1].mean(axis=1)
    cols, new = [], []
    for label, s in (("rv", tgt), ("market", mkt)):
        today = np.zeros(len(s))
        yday = np.zeros(len(s))
        week = np.zeros(len(s))
        prev_means: list = []
        start = 0
        run = 0.0
        last_yday = 0.0
        last_week = 0.0
        for t in range(len(s)):
            if t > 0 and day_id[t] != day_id[t - 1]:
                done = run / (t - start)
                prev_means.append(done)
                last_yday = done
                last_week = float(np.mean(prev_means[-5:]))
                start, run = t, 0.0
            run += s[t]
            today[t] = run / (t - start + 1)
            yday[t] = last_yday
            week[t] = last_week
        cols += [today, yday, week]
        new += [f"{label}_today", f"{label}_yday", f"{label}_week"]
    print(f"[session-memory] appended {new}")
    return (np.ascontiguousarray(np.hstack([X, np.stack(cols, axis=1)])),
            list(names) + new)


def apply_detica(X, names, m=4, tau=48, fit_rows=24000):
    """ANTI-TICA hygiene: project out the m SLOWEST modes of the ladder block.

    TICA orders linear combinations by lag-``tau`` autocorrelation; the top
    modes are the near-unit-root drift we measured to be harmful (time
    proxies). The cap removes drift by assuming it lives in long rungs; this
    removes it wherever it lives, including cross-series combinations. Modes
    are estimated on the first ``fit_rows`` only (all predictions lie beyond)
    and the projection is FROZEN — screen-grade causality, disclosed. The
    cleaned columns keep their names, so cap/geometry knobs stack.
    """
    import re

    idx = [j for j, nm in enumerate(names)
           if re.search(r"_ma_\d+$", nm) and "_x_" not in nm]
    Z = X[:fit_rows, idx]
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Zs = (Z - mu) / sd
    n = len(Zs)
    C0 = Zs.T @ Zs / n
    Ct = (Zs[:-tau].T @ Zs[tau:] + Zs[tau:].T @ Zs[:-tau]) / (2 * (n - tau))
    lam0, U0 = np.linalg.eigh(C0)
    keep = lam0 > lam0.max() * (max(n, C0.shape[0]) * np.finfo(float).eps) ** 2
    T0 = U0[:, keep] / np.sqrt(lam0[keep])[None, :]
    M = T0.T @ Ct @ T0
    M = (M + M.T) / 2
    lam, W = np.linalg.eigh(M)
    Wslow = W[:, ::-1][:, :m]                      # m slowest modes
    Xs_all = (X[:, idx] - mu) / sd
    zw = Xs_all @ T0
    zw_clean = zw - (zw @ Wslow) @ Wslow.T
    Xclean = zw_clean @ np.linalg.pinv(T0) * sd + mu
    X2 = X.copy()
    X2[:, idx] = Xclean
    print(f"[detica] removed {m} slowest modes (top lambdas "
          f"{np.round(lam[::-1][:m], 4)}) from {len(idx)} ladder cols")
    return np.ascontiguousarray(X2), list(names)


def apply_leadlag(X, names, mode="weight", tau=1, fit_rows=24000, npairs=2):
    """LEAD-LAG (Hermitian) machinery on the bar-level series panel.

    Estimates lagged cross-correlations C = B(t)'B(t+tau) on the first
    ``fit_rows`` (frozen, causal for this chunk) and splits C = S + A into
    symmetric co-movement and ANTISYMMETRIC directed flow. A pure common
    trend is symmetric — it leads nothing — so directionality certifies
    dynamics in a way slowness cannot (the detica lesson: the level factor
    is slow AND dynamic; era drift is slow and static).

    mode="weight": scale each series' ladder block by its directionality
    ||A_i|| / ||C_i|| (scale-free, normalized to max 1) — the certificate as
    soft hygiene. mode="feat": append projections of the current
    cross-section onto the top ``npairs`` lead/follow singular axes of A —
    the market's flow state as named features. Prints the leaderboard.
    """
    import re

    cols = [(j, names[j]) for j in range(len(names))
            if re.match(r"^adj_.+_ma_1$", names[j])]
    cols.append((names.index("har_ma_1"), "har_ma_1"))
    idx = [j for j, _ in cols]
    labels = [re.sub(r"^adj_|_ma_1$", "", nm) for _, nm in cols]
    B = X[:fit_rows, idx]
    mu, sd = B.mean(axis=0), B.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Bs = (B - mu) / sd
    n = len(Bs) - tau
    C = Bs[:-tau].T @ Bs[tau:] / n
    A = (C - C.T) / 2
    direction = np.linalg.norm(A, axis=1) / np.maximum(np.linalg.norm(C, axis=1), 1e-300)
    net_lead = A.sum(axis=1)
    order = np.argsort(net_lead)[::-1]
    print("[leadlag] top leaders: " +
          ", ".join(f"{labels[i]}({net_lead[i]:+.2f})" for i in order[:5]))
    print("[leadlag] top followers: " +
          ", ".join(f"{labels[i]}({net_lead[i]:+.2f})" for i in order[-5:]))
    if mode == "weight":
        w = direction / direction.max()
        X2 = X.copy()
        scaled = 0
        for j, lab in zip(idx, labels):
            stem = names[j][:-len("_ma_1")] if names[j] != "har_ma_1" else None
            for k, nm in enumerate(names):
                if stem and nm.startswith(stem + "_ma_"):
                    X2[:, k] = X[:, k] * w[labels.index(lab)]
                    scaled += 1
        print(f"[leadlag] weighted {scaled} ladder cols by directionality "
              f"(range {direction.min():.3f}-{direction.max():.3f})")
        return np.ascontiguousarray(X2), list(names)
    U, sv, Vt = np.linalg.svd(A)
    Ball = (X[:, idx] - mu) / sd
    cols_new, new = [], []
    for p in range(npairs):
        cols_new += [Ball @ U[:, p], Ball @ Vt[p]]
        new += [f"leadflow_u{p+1}", f"leadflow_v{p+1}"]
    print(f"[leadlag] appended {new} (sv {np.round(sv[:npairs], 3)})")
    return (np.ascontiguousarray(np.hstack([X, np.stack(cols_new, axis=1)])),
            list(names) + new)


def chunk_backtest(X, y, baselines, names, fit_predict, hp):
    lo = max(0, START - HALO)
    Xc, yc, bc = X[lo:END], y[lo:END], baselines[lo:END]
    hp = dict(hp, _feature_names=list(names))
    preds = fit_predict(Xc, yc, TRAIN_WIN, hp)
    y_oos, b_oos = yc[TRAIN_WIN:], bc[TRAIN_WIN:]
    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, b_oos)
    return pred_raw, true_raw


def main() -> None:
    arms = []
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        tag, geom, w = parts[0], parts[1], int(parts[2])
        k = int(parts[3]) if len(parts) > 3 else 25
        d = int(parts[4]) if len(parts) > 4 else 0
        resid = parts[5] if len(parts) > 5 else "identity"
        decay = float(parts[6]) if len(parts) > 6 else 1.0
        cap = int(parts[7]) if len(parts) > 7 else 0
        loc = float(parts[8]) if len(parts) > 8 else 0.0
        navma = int(parts[9]) if len(parts) > 9 else 0
        bands = int(parts[10]) if len(parts) > 10 else 0
        omega = float(parts[11]) if len(parts) > 11 else 1.0
        arms.append((tag, geom, w, k, d, resid, decay, cap, loc, navma, bands, omega))
    if not arms:
        arms = [("euclid12", "euclid", 12, 25, 0, "identity", 1.0, 0, 0.0, 0, 0, 1.0)]

    # incumbent: exact OLS on HAR+calendar, empty-exog bucket, per-bar refit —
    # the shared comparator of every audited table, on the same chunk.
    Xb, yb, bb, nb = prepare_full("baseline")
    t0 = time.time()
    p_i, t_i = chunk_backtest(
        Xb, yb, bb, nb, fit_predict_ridge,
        dict(alpha=0.0, _refit_frequency=1, _incremental=True),
    )
    print(f"[incumbent] qlike={forecast_metrics(p_i, t_i)['qlike']:.6f} "
          f"({time.time() - t0:.0f}s)")

    X, y, b, names = prepare_full(BUCKET)
    RESID = {
        "identity": dict(residualizer_kind="identity"),
        "ridge1": dict(residualizer_kind="ridge", ridge_alpha=1.0),
        "ols0": dict(residualizer_kind="ridge", ridge_alpha=0.0),
    }
    rows = []
    Xb_cache = None
    for tag, geom, w, k, d, resid, decay, cap, loc, navma, bands, omega in arms:
        hp = dict(
            RESID[resid], view_exog=True, geometry=geom,
            geometry_tau=PERIODS_PER_DAY, ladder_base=2, view_window=w,
            embedding_dim=d, graph_k=10, neighbor_k=k, refit_frequency=240,
            omega=omega, seed=42, pls_decay=decay, state_rung_cap=cap,
            local_alpha=loc, drop_avail_ma=bool(navma),
            fast_anchor=bool(int(os.environ.get("FAST_ANCHOR", "0"))),
            stack_window=int(os.environ.get("STACK_WINDOW", "0")),
        )
        # bands field: 0 = none, 1 = band-pass reparam, 2 = scale-poly deg 2,
        # 3 = scale-poly deg 1 (level+slope), 4 = full ladder + shock ages,
        # 5 = scale-poly deg 1 + shock ages (the echo-hypothesis test)
        names_arm = names
        if bands == 1:
            if Xb_cache is None:
                Xb_cache = (apply_bands(X, names), names)
            X_arm, names_arm = Xb_cache
        elif bands in (2, 3):
            X_arm, names_arm = apply_scale_poly(X, names, cap=cap or 256,
                                                deg=2 if bands == 2 else 1)
            hp["state_rung_cap"] = 0  # cap already embedded in the transform
        elif bands in (4, 5):
            X_arm, names_arm = apply_shock(X, names)
            if bands == 5:
                X_arm, names_arm = apply_scale_poly(X_arm, names_arm,
                                                    cap=cap or 256, deg=1)
                hp["state_rung_cap"] = 0
        elif bands in (6, 7):
            X_arm, names_arm = apply_target_pol(X, names, cap=cap or 256,
                                                quad=(bands == 7))
        elif bands == 8:
            X_arm, names_arm = apply_maxladder(X, names)
        elif bands == 9:
            X_arm, names_arm = apply_volofvol(X, names)
        elif bands == 10:
            X_arm, names_arm = apply_session_memory(X, names)
        elif bands in (11, 12):
            X_arm, names_arm = apply_detica(X, names, m=(4 if bands == 11 else 16))
        elif bands == 13:
            X_arm, names_arm = apply_leadlag(
                X, names, mode="weight", tau=int(os.environ.get("LL_TAU", "1")))
        elif bands == 14:
            X_arm, names_arm = apply_leadlag(
                X, names, mode="feat", tau=int(os.environ.get("LL_TAU", "1")))
        else:
            X_arm = X
        t0 = time.time()
        p_t, t_t = chunk_backtest(X_arm, y, b, names_arm,
                                  fit_predict_spectral_knn, hp)
        # persist-raw-outputs ruling: per-bar predictions, not just derived
        # metrics — arm-vs-arm DM needs the pooled per-bar series.
        os.makedirs(os.path.join(CACHE_DIR, "geo_preds"), exist_ok=True)
        np.savez(os.path.join(CACHE_DIR, "geo_preds", f"{tag}.npz"),
                 pred_raw=p_t, true_raw=t_t, incumbent_pred_raw=p_i)
        m = forecast_metrics(p_t, t_t, benchmark=p_i)
        dm = dm_test(qlike_per_bar(p_t, t_t), qlike_per_bar(p_i, t_t), h=HORIZON)
        row = dict(tag=tag, geometry=geom, view_w=w, neighbor_k=k, d=d,
                   resid=resid, cap=cap, qlike=m["qlike"], oos_r2=m["oos_r2"],
                   mz_beta=m["mz_beta"], dm=dm["dm"], dm_p=dm["p"], n=m["n"],
                   secs=round(time.time() - t0))
        rows.append(row)
        print(row)
    out = pd.DataFrame(rows)
    out["incumbent_qlike"] = forecast_metrics(p_i, t_i)["qlike"]
    path = os.path.join(CACHE_DIR, "geometry_chunk_battery.csv")
    header = not os.path.exists(path)
    out.to_csv(path, mode="a", header=header, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
