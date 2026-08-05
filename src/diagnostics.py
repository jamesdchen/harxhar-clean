"""Feature-health invariants, checked at panel-build time.

Ten defects of the same class have now been found in this pipeline, every one of them by
*symptom* — a blown QLIKE days later, usually after a cluster round-trip — and every one of them
detectable in seconds by a cheap invariant on the feature itself. The list, and the check that
catches each, is in ``analysis/diagnostics_backtest.py``; a check that does not fire on its own
historical bug is not worth having, so each one here is validated against the incident that
motivated it.

These are **invariant tests, not performance comparisons**: they never look at forecast accuracy,
so they consume no inferential budget and can run on every build forever (see the writeup's §10
selection audit for why that distinction matters here).

Five checks, each with a documented incident behind it:

1. :func:`divisor_health` — **pinned-divisor fraction.** The share of rows where
   ``diurnal_adjust``'s per-slot std divisor sits at its floor. At 1.0 the "adjustment" is
   division by a global constant and cannot track the feature's scale. `voldemand` ran at 0.74
   undetected for a whole campaign.
2. :func:`column_health` — **tail profile** (max / p99.9 / p99.99 of |z|). Panel-wide p99.9 is
   ~7.6, so a column at 400 is not a fat tail, it is a broken transform.
3. :func:`column_health` — **era asymmetry**: max |z| in the last third over the first third.
   The single fastest voldemand detector (51x) and the one that specifically catches scale drift
   rather than static fat tails.
4. :func:`column_health` — **modal share**: the frequency of the single most common value. Catches
   both a wrong-constant fill (the ``apply_overnight_fills`` 1.0 artifact, which manufactured a
   robust-looking 16/16-year predictor out of missingness structure) and un-hurdled
   zero-inflation.
5. :func:`guard_health` — **scale-guard bind rate**: how often the causal reference IQR has to
   override the local window IQR in ``rolling_robust_scale``. This is the Part-3
   divide-by-a-degenerate-scale class expressed as a number.

Plus one run-level check, :func:`prediction_health`, which would have flagged the R² −0.635 arm of
the interaction study the moment it was produced instead of two turns later.

Thresholds are deliberately few and derived from the incidents rather than tuned; a check that
cries wolf gets muted, and then it is worse than nothing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.features.transforms.target import DIURNAL_PINNED_MAX, PERIODS_PER_DAY, diurnal_adjust

# Thresholds. Each is anchored to a measured incident, not chosen for aesthetics.
MAX_ABS_Z_FAIL = 50.0  # voldemand hit 2011; the healthy panel's p99.9 is 7.6
MAX_ABS_Z_WARN = 20.0
ERA_ASYM_FAIL = 5.0  # voldemand 51x (487 in 2021-23 vs 9.6 in 2011-16)
ERA_ASYM_WARN = 2.5
# Era asymmetry is a ratio of maxima, so it is meaningless when both maxima are small: post-fix
# voldemand goes 1.9 -> 3.2 across eras, a 1.7x "asymmetry" on a perfectly healthy column. The
# back-test flagged the fixed column on exactly this before the level gate was added. Asymmetry is
# only evidence of drift if the late-era tail is itself non-trivial.
ERA_ASYM_MIN_LEVEL = 10.0
PINNED_FAIL = DIURNAL_PINNED_MAX  # 0.5, the same gate robust_transform acts on
PINNED_WARN = 0.2  # sentiment 0.28, sumret3_vwstock 0.27 sit here
MODAL_SHARE_WARN = 0.30  # voldemand's raw zero mass is 0.61; the 1.0-fill artifact was ~0.3+
GUARD_BIND_WARN = 0.05
PRED_SCALE_FAIL = 10.0  # |pred| / sd(y); the -0.635 arm produced 7.3 against sd 0.249


def _status(row: dict) -> str:
    """Worst-of over the individual checks."""
    drifting = row["max_abs_z"] >= ERA_ASYM_MIN_LEVEL  # gate: see ERA_ASYM_MIN_LEVEL
    # For a hurdle-encoded column the tail is judged on p99.9 rather than on the max. A genuine point
    # mass makes the max uninformative: ``numobs`` is capped at 30 and is 30 on 78% of bars, so its
    # IQR is a rounding artifact and an ordinary holiday half-session is correctly ~79 sigma away —
    # no scale estimator makes that number small without misdescribing the distribution. The
    # extensive margin is what carries the information and the ``_active`` indicator now carries it
    # explicitly; ``p99.9`` still catches a broken *transform* (``adj_numobs_ma_25`` sits at 46, so
    # it stays a warn, which is the correct verdict rather than a muted one). Same principle as the
    # modal-share exemption below: a point mass is only a defect if nothing is modelling it.
    tail = row["p99.9"] if row.get("hurdle_encoded", False) else row["max_abs_z"]
    if (
        tail >= MAX_ABS_Z_FAIL
        or (drifting and row["era_asymmetry"] >= ERA_ASYM_FAIL)
        or row.get("pinned_divisor_frac", 0.0) >= PINNED_FAIL
    ):
        return "FAIL"
    # A point mass is only a defect if nothing is modelling it. Zero-inflated columns already get
    # the hurdle encoding (an ``_active`` occurrence indicator plus an active-values scale), so
    # their modal share is by design, not a finding — the back-test flagged post-fix ``voldemand``
    # on this before the exemption was added.
    modal_is_defect = row["modal_share"] >= MODAL_SHARE_WARN and not row.get("hurdle_encoded", False)
    if (
        tail >= MAX_ABS_Z_WARN
        or (drifting and row["era_asymmetry"] >= ERA_ASYM_WARN)
        or row.get("pinned_divisor_frac", 0.0) >= PINNED_WARN
        or modal_is_defect
        or row.get("guard_bind_rate", 0.0) >= GUARD_BIND_WARN
    ):
        return "warn"
    return "ok"


def column_health(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Checks 2-4 on a finished (scaled) feature matrix: tails, era asymmetry, modal share.

    ``era_asymmetry`` compares the last third of the sample to the first third, so it isolates
    *drift-induced* tails from a column that is merely fat-tailed throughout.
    """
    n = len(X)
    a = np.abs(X)
    third = n // 3
    early = a[:third].max(axis=0)
    late = a[-third:].max(axis=0)
    rows = []
    for j, name in enumerate(names):
        col = X[:, j]
        # modal share via a fine histogram: exact-value counting is meaningless for floats, but a
        # genuine point mass (a constant fill, a zero mass) concentrates in one narrow bin.
        finite = col[np.isfinite(col)]
        if finite.size:
            lo, hi = np.percentile(finite, (0.5, 99.5))
            if hi > lo:
                counts, _ = np.histogram(finite, bins=200, range=(lo, hi))
                modal = float(counts.max() / finite.size)
            else:
                modal = 1.0
        else:
            modal = np.nan
        rows.append(
            {
                "feature": name,
                "max_abs_z": float(a[:, j].max()),
                "p99.9": float(np.percentile(a[:, j], 99.9)),
                "p99.99": float(np.percentile(a[:, j], 99.99)),
                "era_asymmetry": float(late[j] / early[j]) if early[j] > 0 else np.inf,
                "modal_share": modal,
            }
        )
    return pd.DataFrame(rows)


def divisor_health(
    df: pd.DataFrame, cols: list[str], time_col: str = "time_of_day"
) -> pd.DataFrame:
    """Check 1: the fraction of rows whose per-slot std divisor is pinned at its floor.

    Only signed features take the std path in :func:`diurnal_adjust`; unsigned ones divide by a
    per-slot rolling *mean*, whose analogous degeneracy is the ``replace(0, 1.0)`` branch.

    **Reported as 0 when the divide is not in the pipeline.** ``robust_transform`` routes a column
    past ``diurnal_adjust`` entirely once the pin rate crosses ``DIURNAL_PINNED_MAX`` (it uses
    :func:`~src.features.transforms.target.asinh_stabilize` instead), so measuring the pin rate of
    a divisor that no longer runs is a false alarm — the back-test caught exactly that, flagging
    post-fix ``voldemand`` as still broken. ``divide_in_use`` records which branch applies.
    """
    from src.features.transforms.target import has_name_stabilizer, is_diurnal_excluded

    rows = []
    for col in cols:
        s = df[col]
        signed = bool((s.dropna() < 0).any())
        # A column ``robust_transform`` routes past the diurnal entirely has no divisor to measure.
        # Same false-alarm class the ``asinh`` branch produced: reporting the pin rate of a divisor
        # that never runs. ``vix`` / ``vvix`` / ``vix3m`` / ``stocktwits_sentiment`` are here.
        if is_diurnal_excluded(col):
            rows.append(
                {"feature": col, "signed": signed, "divide_in_use": False,
                 "pinned_divisor_frac": 0.0}
            )
            continue
        _, base = diurnal_adjust(s, df[time_col], signed)
        pinned = float(
            np.isclose(base, base.min() if signed else 1.0, rtol=1e-9 if signed else 1e-12).mean()
        )
        stabilized = signed and not has_name_stabilizer(col) and pinned >= DIURNAL_PINNED_MAX
        rows.append(
            {
                "feature": col,
                "signed": signed,
                "divide_in_use": not stabilized,
                "pinned_divisor_frac": 0.0 if stabilized else pinned,
            }
        )
    return pd.DataFrame(rows)


def guard_health(
    X: np.ndarray, ref_iqr: np.ndarray | None, train_win: int, grid: int = 4800
) -> pd.DataFrame:
    """Check 5: how often the causal reference IQR has to override the local window IQR.

    A high bind rate means the local trailing window is *routinely* degenerate for this column —
    i.e. without the guard the scaler would be dividing by ~0, which is the Part-3 failure. The
    local IQR is evaluated on a coarse grid because the quantity is slowly varying and an exact
    per-row recomputation would cost more than the panel build.
    """
    if ref_iqr is None:
        return pd.DataFrame({"guard_bind_rate": []})
    n, p = X.shape
    binds = np.zeros(p)
    checks = 0
    for t in range(train_win, n, grid):
        w = X[t - train_win : t]
        q25, q75 = np.percentile(w, (25.0, 75.0), axis=0)
        local = q75 - q25
        binds += (ref_iqr[t] > local).astype(float)
        checks += 1
    return pd.DataFrame({"guard_bind_rate": binds / max(checks, 1)})


def prediction_health(y: np.ndarray, pred: np.ndarray) -> dict:
    """Run-level sanity: a forecast whose scale dwarfs the target's is broken, not merely wrong."""
    sd = float(np.std(y)) or 1.0
    ratio = float(np.max(np.abs(pred[np.isfinite(pred)])) / sd)
    return {
        "max_abs_pred_over_sd_y": ratio,
        "status": "FAIL" if ratio >= PRED_SCALE_FAIL else "ok",
    }


def feature_health_report(
    X: np.ndarray,
    names: list[str],
    df: pd.DataFrame | None = None,
    raw_cols: list[str] | None = None,
    ref_iqr: np.ndarray | None = None,
    train_win: int = 250 * PERIODS_PER_DAY,
) -> pd.DataFrame:
    """Assemble all five checks into one per-column report with an ok / warn / FAIL status.

    ``df`` + ``raw_cols`` enable check 1 (it needs the pre-transform series); ``ref_iqr`` enables
    check 5. Both are optional so the report still runs on a bare matrix.
    """
    rep = column_health(X, names)
    if df is not None and raw_cols:
        dh = divisor_health(df, raw_cols)
        # map a raw column's divisor diagnostic onto every feature derived from it
        stem = {c: c for c in raw_cols}
        rep["raw"] = [
            next((c for c in stem if c in n), "") for n in rep["feature"]
        ]
        rep = rep.merge(
            dh.rename(columns={"feature": "raw"})[["raw", "pinned_divisor_frac"]],
            on="raw",
            how="left",
        )
    if ref_iqr is not None:
        rep["guard_bind_rate"] = guard_health(X, ref_iqr, train_win)["guard_bind_rate"].to_numpy()
    rep["status"] = [_status(r) for _, r in rep.iterrows()]
    return rep.sort_values(
        ["status", "max_abs_z"], ascending=[True, False], key=lambda s: s.map(
            {"FAIL": 0, "warn": 1, "ok": 2}
        ) if s.name == "status" else s
    )


# ---------------------------------------------------------------------------
# Build-time hook
# ---------------------------------------------------------------------------

STRICT_ENV = "FEATURE_HEALTH_STRICT"


def _hurdle_flags(names: list[str], df: pd.DataFrame | None) -> list[bool]:
    """Which columns descend from a raw exog that got the hurdle (``_active``) encoding.

    Needed by :func:`_status`: a modal share is by design for these, not a defect.
    """
    if df is None:
        return [False] * len(names)
    stems = [c[: -len("_active")] for c in df.columns if c.endswith("_active")]
    return [any(s in n for s in stems) for n in names]


def check_and_report(
    X: np.ndarray,
    names: list[str],
    *,
    df: pd.DataFrame | None = None,
    raw_cols: list[str] | None = None,
    ref_iqr: np.ndarray | None = None,
    train_win: int = 250 * PERIODS_PER_DAY,
    output_path: str | None = None,
    label: str = "panel",
    strict: bool | None = None,
    top: int = 10,
) -> pd.DataFrame:
    """Run every invariant on a finished feature matrix, print a summary, optionally hard-fail.

    This is the entry point the *build* calls, so the checks run on every panel that is ever
    constructed rather than when somebody remembers to look. Ten defects of this class have been
    found by symptom days later; none of them would have survived this being on.

    ``strict`` decides what a FAIL does. It defaults to the ``FEATURE_HEALTH_STRICT`` environment
    variable, and that default is **off**, deliberately: the panel as it stands today has 26
    columns at FAIL (``adj_sumpret2_vwstock_ma_5`` at |z| 82, ``adj_numobs_ma_25`` at 79, the
    ``stocktwits_sentiment`` family at 70), so a hard raise would stop every run in the repo. Off,
    the report is written and the count is printed on every build, which is the part that was
    missing. Turn it on per-run (``FEATURE_HEALTH_STRICT=1``) to make a regression fatal once the
    standing failures are cleared.
    """
    if strict is None:
        strict = os.environ.get(STRICT_ENV, "").lower() in ("1", "true", "yes")
    rep = feature_health_report(
        X, names, df=df, raw_cols=raw_cols, ref_iqr=ref_iqr, train_win=train_win
    )
    rep["hurdle_encoded"] = _hurdle_flags(list(rep["feature"]), df)
    rep["status"] = [_status(r) for _, r in rep.iterrows()]
    counts = rep["status"].value_counts().to_dict()
    n_fail = int(counts.get("FAIL", 0))
    print(
        f"[feature-health/{label}] {len(rep)} cols: "
        f"{counts.get('ok', 0)} ok / {counts.get('warn', 0)} warn / {n_fail} FAIL",
        flush=True,
    )
    worst = rep[rep["status"] != "ok"].head(top)
    for _, r in worst.iterrows():
        print(
            f"[feature-health/{label}]   {r['status']:>4s} {r['feature']:<40s} "
            f"max|z| {r['max_abs_z']:7.1f}  era {r['era_asymmetry']:6.2f}  "
            f"modal {r['modal_share']:.3f}",
            flush=True,
        )
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        rep.to_csv(output_path, index=False)
        print(f"[feature-health/{label}] wrote {output_path}", flush=True)
    if strict and n_fail:
        raise ValueError(
            f"feature-health: {n_fail} column(s) FAIL an invariant "
            f"({', '.join(rep[rep['status'] == 'FAIL']['feature'].head(5))}...). "
            f"Unset {STRICT_ENV} to downgrade to a warning."
        )
    return rep
