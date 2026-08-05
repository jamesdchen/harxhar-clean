"""§15.3 item 4: the twelve-plus columns the invariants flag at max|z| 68-82.

``src/diagnostics.py`` now runs on every build, and on the current panel it fails 26 columns.
``voldemand`` (§8) was the worst and is fixed; these are what is left. They are *not* one disease,
and the point of this module is to establish which mechanism each family has before changing
anything — the §8 precedent is that the first plausible story ("self-inflicted clip") was wrong and
only a per-row trace of where the extreme value comes from settled it.

The three candidate mechanisms, and the evidence that would distinguish them:

1. **Degenerate per-slot divisor** (the ``voldemand`` disease, milder). A signed column divides by
   a per-slot rolling std whose floor binds on a minority of rows; on those rows the "adjustment"
   is division by a global constant. Signature: ``pinned_divisor_frac`` in 0.2-0.5 (below the gate
   that already routes a column to ``asinh_stabilize``), extreme concentrated in single rows,
   ``p99.9`` normal. Predicted family: ``stocktwits_sentiment`` (max = p99.99 = 70.34 exactly, so
   one row, and its era asymmetry is 0.05 — the spike is *early*).
2. **Unwinsorized warm-up / NaN-holed window.** ``rolling_winsorize`` uses ``min_periods=window``,
   so its clip bounds are NaN for the first 240 rows *and* for any window containing a NaN — and
   ``Series.clip`` with NaN bounds is a no-op. A column with sparse early coverage is therefore
   completely unwinsorized exactly where it is worst behaved. Signature: extreme in the first
   third, at or near the start of the column's coverage.
3. **Long-MA imputation seam.** ``_build_scale_guards`` floors the scale of ``adj_<col>_ma_w`` by
   the expanding IQR over rows where *the raw column* is available. For ``w = 3125`` (~65 days)
   the MA is still mostly imputed-median for thousands of rows after availability starts, so the
   guard's notion of "real" is wrong and the level jumps at the seam. Signature: only the long
   windows fail, and ``era_asymmetry`` is infinite (early-third max exactly 0 = flat imputed
   constant). Predicted family: ``vvix``, ``vix3m`` at ``ma_3125``.

A fourth possibility — the column is simply fat-tailed and the transform is fine — is the null, and
it is what ``p99.9`` tests: the panel's p99.9 is ~7.6, so a column with p99.9 of 46
(``adj_numobs_ma_25``) has a broken *distribution*, whereas one at p99.9 of 7 and max 82 has a
handful of broken *rows*.

Stages
------
``diagnose``
    Per flagged column: the raw column's shape, the semantic rule, the pinned-divisor fraction, the
    max |z| and *where* it lands (date, and which era), whether the winsorizer's bounds were NaN
    there, and whether the scale guard's reference IQR was zero there. No changes.
``fix``
    Apply the mechanism-specific fixes and re-measure. Reports, per column, the before/after tails
    and — the §8 acceptance test — how many *unrelated* columns move at all. A fix that perturbs
    healthy columns is rejected.
``verify``
    Full-panel invariant report before vs after, plus the forecast-level control: the HAR+exog
    walk-forward R2 on the affected buckets. A data-quality fix is justified by the invariant, not
    by the score, so this is a *guard against damage*, not a success criterion.

Usage
-----
    python analysis/tail_fix.py --stage diagnose
    python analysis/tail_fix.py --stage fix
    python analysis/tail_fix.py --stage verify
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.executor import (  # noqa: E402
    ZERO_INFLATED_FRAC,
    _build_scale_guards,
    mode_inflated,
)
from src.data.loading import ALL_FEATURES, load_raw_data  # noqa: E402
from src.features.extractors.har import generate_har_features, resolve_har_lags  # noqa: E402
from src.features.transforms.scaling import rolling_robust_scale  # noqa: E402
from src.features.transforms.target import (  # noqa: E402
    PERIODS_PER_DAY,
    WINSOR_LOWER_Q,
    diurnal_adjust,
    has_name_stabilizer,
    robust_transform,
)

OUT = "results/alpha_manifestation"
TRAIN_WINDOW_DAYS = 250
WINSOR_WINDOW = 240

# The FAIL set from ``feature_health_production.csv``, reduced to raw columns. Kept explicit
# rather than re-derived from a threshold so the target set cannot drift between stages.
FLAGGED_RAW = [
    "sumpret2_vwstock",
    "sumpret2_ewstock",
    "sumbipow_ewstock",
    "sumbipow_vwstock",
    "sumret2_vwstock",
    "sumret2_ewstock",
    "sumret4_vwstock",
    "sumret4_ewstock",
    "sumabsret_ewstock",
    "numobs",
    "stocktwits_sentiment",
    "vix3m",
    "vvix",
]


def _load() -> pd.DataFrame:
    df = load_raw_data("data", allow_missing=True)
    df[list(ALL_FEATURES)] = df[list(ALL_FEATURES)].ffill()
    return df.dropna(subset=["RV"]).reset_index(drop=True)


def _rule(col: str, signed: bool) -> str:
    name = col.lower()
    if any(t in name for t in ("ret2", "rv", "turnover", "bipow", "effspread")):
        return "1 sqrt"
    if "autocov" in name:
        return "2 signed-sqrt"
    if "ret3" in name:
        return "3 cbrt"
    if "ret4" in name:
        return "4 fourth-root"
    if signed or "sumabsret" in name:
        return "5 identity"
    return "6 log"


def stage_diagnose() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    df = _load()
    n = len(df)
    print(f"{n} bars, {df['t'].iloc[0]} .. {df['t'].iloc[-1]}\n", flush=True)

    rows = []
    for col in FLAGGED_RAW:
        raw = df[col]
        fin = raw.dropna()
        signed = bool((fin < 0).any())
        adj, base = robust_transform(df, col, use_transform=True, use_diurnal=True)
        _, base_raw = diurnal_adjust(raw, df["time_of_day"], signed)
        pinned = float(np.isclose(base_raw, base_raw.min(), rtol=1e-9).mean()) if signed else 0.0

        # where the winsorizer had no bounds: a NaN in the trailing window (or the warm-up)
        lo = raw.rolling(WINSOR_WINDOW, min_periods=WINSOR_WINDOW).quantile(WINSOR_LOWER_Q).shift(1)
        unwins = float(lo.isna().mean())
        first_real = int(raw.notna().to_numpy().argmax()) if raw.notna().any() else -1

        rows.append(
            {
                "raw": col,
                "n_nan": int(raw.isna().sum()),
                "nan_frac": float(raw.isna().mean()),
                "zero_frac": float((fin == 0).mean()) if len(fin) else np.nan,
                "signed": signed,
                "rule": _rule(col, signed),
                "has_name_stabilizer": has_name_stabilizer(col),
                "pinned_divisor_frac": pinned,
                "unwinsorized_frac": unwins,
                "first_real_row": first_real,
                "first_real_date": str(df["t"].iloc[first_real])[:10] if first_real >= 0 else "",
                "raw_max": float(fin.abs().max()) if len(fin) else np.nan,
                "raw_p99.9": float(fin.abs().quantile(0.999)) if len(fin) else np.nan,
                "adj_max": float(adj.abs().max()),
                "adj_p99.9": float(adj.abs().quantile(0.999)),
            }
        )
        df[f"adj_{col}"] = adj
    rep = pd.DataFrame(rows)
    print("RAW / TRANSFORM SHAPE")
    print(rep.drop(columns=["n_nan", "first_real_row"]).to_string(index=False), flush=True)

    # Now the scaled panel, restricted to the flagged families, so max |z| can be traced to a row.
    for col in FLAGGED_RAW:
        s = df[f"adj_{col}"]
        df[f"adj_{col}"] = s.fillna(s.median())
        df[f"{col}_avail"] = df[col].notna().astype("float64")
        v = df[col].to_numpy(dtype=np.float64)
        fin = v[~np.isnan(v)]
        if fin.size and float(np.mean(fin == 0.0)) >= ZERO_INFLATED_FRAC:
            df[f"{col}_active"] = (np.nan_to_num(v) != 0.0).astype("float64")
    adj_cols = [f"adj_{c}" for c in FLAGGED_RAW] + [f"{c}_avail" for c in FLAGGED_RAW]
    adj_cols += [f"{c}_active" for c in FLAGGED_RAW if f"{c}_active" in df.columns]
    d2, har_names = generate_har_features(df, target_col="RV", exog_cols=adj_cols)
    keep = [c for c in har_names if any(f"adj_{r}_ma_" in c for r in FLAGGED_RAW)]
    d2 = d2.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = d2[keep].to_numpy(dtype=np.float64)
    ref_iqr, fixed = _build_scale_guards(X, d2, keep)
    Z = rolling_robust_scale(
        X, TRAIN_WINDOW_DAYS * PERIODS_PER_DAY, ref_iqr=ref_iqr, fixed_cols=fixed
    )
    third = len(Z) // 3
    print("\nWHERE THE EXTREME LANDS (scaled panel, flagged families only)")
    trace = []
    for j, name in enumerate(keep):
        a = np.abs(Z[:, j])
        i = int(np.nanargmax(a))
        if a[i] < 20:
            continue
        era = "early" if i < third else ("mid" if i < 2 * third else "late")
        trace.append(
            {
                "feature": name,
                "max_abs_z": float(a[i]),
                "p99.9": float(np.nanpercentile(a, 99.9)),
                "argmax_date": str(d2["t"].iloc[i])[:16],
                "era": era,
                "n_above_20": int((a > 20).sum()),
                "ref_iqr_at_argmax": float(ref_iqr[i, j]) if ref_iqr is not None else np.nan,
                "raw_x_at_argmax": float(X[i, j]),
            }
        )
    tr = pd.DataFrame(trace).sort_values("max_abs_z", ascending=False)
    print(tr.to_string(index=False), flush=True)
    rep.to_csv(f"{OUT}/tailfix_diagnose_raw.csv", index=False)
    tr.to_csv(f"{OUT}/tailfix_diagnose_trace.csv", index=False)
    print(f"\nwrote {OUT}/tailfix_diagnose_raw.csv + tailfix_diagnose_trace.csv")


def stage_trace() -> None:
    """Follow one bar through the whole chain for each family, because the diagnosis is not obvious.

    ``diagnose`` refuted two of the three predicted mechanisms for the ``sum*stock`` family: the
    per-slot divisor is not pinned (0.000) and the winsorizer is not blind there (0.001 of rows).
    Yet the *adjusted, winsorized, pre-MA* series already reaches 46 against a p99.9 of 0.0005 in
    raw units — and every argmax lands on an overnight or post-holiday bar (2024-01-17 03:00,
    2023-05-01 03:30, 2023-07-04 14:00, and ``numobs`` on 2014-12-25). So the question is narrow:
    at those rows, which factor is extreme — the raw print, the per-slot baseline, or the
    winsorizer's bound?
    """
    df = _load()
    tod = df["time_of_day"]
    for col, when in (
        ("sumpret2_vwstock", "2024-01-17 03:00"),
        ("sumbipow_ewstock", "2024-01-17 03:00"),
        ("sumabsret_ewstock", "2023-05-01 03:30"),
        ("numobs", "2014-12-25 16:30"),
        ("stocktwits_sentiment", "2010-06-02 05:30"),
    ):
        raw = df[col]
        signed = bool((raw.dropna() < 0).any())
        adjusted, base = diurnal_adjust(raw, tod, signed)
        from src.features.transforms.target import apply_semantic_transform, rolling_winsorize

        sem = apply_semantic_transform(adjusted, col, signed)
        wins = rolling_winsorize(sem, window=WINSOR_WINDOW)
        up = sem.rolling(WINSOR_WINDOW, min_periods=WINSOR_WINDOW).quantile(0.95).shift(1)
        i = int((df["t"] == pd.Timestamp(when)).to_numpy().argmax())
        slot = tod.iloc[i]
        in_slot = raw[tod == slot]
        print(f"\n{col} @ {when}  (slot {slot}, {len(in_slot)} in-slot bars)")
        print(f"  raw x            {raw.iloc[i]:.6g}   in-slot median {in_slot.median():.6g}  "
              f"in-slot p99 {in_slot.quantile(0.99):.6g}  global p99.9 {raw.abs().quantile(0.999):.6g}")
        print(f"  diurnal baseline {base.iloc[i]:.6g}   (in-slot baseline median "
              f"{base[tod == slot].median():.6g}, ratio {base.iloc[i] / base[tod == slot].median():.4g})")
        print(f"  adjusted         {adjusted.iloc[i]:.6g}")
        print(f"  semantic         {sem.iloc[i]:.6g}")
        print(f"  winsor upper     {up.iloc[i]:.6g}   -> winsorized {wins.iloc[i]:.6g}")
        print(f"  overnight bar?   {slot < pd.Timestamp('1900-01-01 09:30').time() or slot > pd.Timestamp('1900-01-01 16:00').time()}")


def stage_modes() -> None:
    """Modal share of every raw exog, to see whether ``numobs`` is a family or a singleton.

    ``numobs`` is capped at 30 and *is* 30 on essentially every bar (raw max = raw p99.9 = 30), so
    its IQR is ~0 and any real dip becomes |z| ~ 80. That is not a broken transform, it is a column
    whose scale is undefined — the same shape the repo already handles for zero-inflated features
    with the hurdle encoding, except the point mass is at the mode rather than at zero. Whether the
    fix should generalize ``ZERO_INFLATED_FRAC`` to a mode test depends on how many columns look
    like this, which is what this measures.
    """
    df = _load()
    rows = []
    for col in ALL_FEATURES:
        if col not in df.columns:
            continue
        v = df[col].dropna()
        if not len(v):
            continue
        vc = v.value_counts()
        rows.append(
            {
                "raw": col,
                "modal_value": float(vc.index[0]),
                "modal_share": float(vc.iloc[0] / len(v)),
                "zero_share": float((v == 0).mean()),
                "n_unique": int(v.nunique()),
                "iqr": float(v.quantile(0.75) - v.quantile(0.25)),
                "p99_minus_p1": float(v.quantile(0.99) - v.quantile(0.01)),
            }
        )
    d = pd.DataFrame(rows).sort_values("modal_share", ascending=False)
    print(d.to_string(index=False))
    d.to_csv(f"{OUT}/tailfix_modal_shares.csv", index=False)
    print(f"\nwrote {OUT}/tailfix_modal_shares.csv")


# ---------------------------------------------------------------------------
# The fixes, measured one at a time
# ---------------------------------------------------------------------------

# Each arm is (label, slot_band, stem_exclusion, mode_hurdle, guard_fix). ``pre`` must reproduce the
# recorded pre-fix state or the whole ladder is meaningless — that assertion is not decoration, it is
# the check that caught ``voldemand_fix.build_vd_block("status_quo")`` silently drifting in §8.
ARMS: list[tuple[str, float, bool, bool, bool]] = [
    ("pre", 0.0, False, False, False),
    ("band0.50", 0.50, False, False, False),
    ("band0.25", 0.25, False, False, False),
    ("band0.10", 0.10, False, False, False),
    ("band+excl", 0.25, True, False, False),
    ("band+excl+mode", 0.25, True, True, False),
    ("all", 0.25, True, True, True),
]
# The pre-fix max |z| of the worst column, from ``feature_health_production.csv``. Reproduced to
# within float32 storage error or the ladder is rejected.
PRE_FIX_WORST = ("adj_sumpret2_vwstock_ma_5", 82.1)


def _build_flagged(
    slot_band: float, stem_excl: bool, mode_hurdle: bool, guard_fix: bool
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Build the scaled panel restricted to the flagged families under one fix configuration.

    Mirrors ``alpha_panel.build_panel`` exactly for the columns involved (transform ->
    impute-and-indicate -> hurdle -> HAR -> scale guards -> rolling robust scale), with each fix
    behind a switch so the ladder attributes the change to a mechanism rather than to "the new code".
    """
    import src.features.transforms.target as tgt

    df = _load()
    saved = tgt.DIURNAL_EXCLUDED_STEMS
    if not stem_excl:
        tgt.DIURNAL_EXCLUDED_STEMS = ()
    try:
        adj_cols: list[str] = []
        for col in FLAGGED_RAW:
            adj, _ = tgt.robust_transform(
                df, col, use_transform=True, use_diurnal=True, slot_band=slot_band
            )
            df[f"adj_{col}"] = adj
            adj_cols.append(f"adj_{col}")
    finally:
        tgt.DIURNAL_EXCLUDED_STEMS = saved

    for col in FLAGGED_RAW:
        s = df[f"adj_{col}"]
        df[f"adj_{col}"] = s.fillna(s.median())
        df[f"{col}_avail"] = df[col].notna().astype("float64")
        adj_cols.append(f"{col}_avail")
    for col in FLAGGED_RAW:
        v = df[col].to_numpy(dtype=np.float64)
        if mode_hurdle:
            inflated, mode = mode_inflated(v)
        else:
            fin = v[~np.isnan(v)]
            inflated = bool(fin.size and float(np.mean(fin == 0.0)) >= ZERO_INFLATED_FRAC)
            mode = 0.0
        if inflated:
            df[f"{col}_active"] = (np.nan_to_num(v, nan=mode) != mode).astype("float64")
            adj_cols.append(f"{col}_active")

    d2, har_names = generate_har_features(df, target_col="RV", exog_cols=adj_cols)
    keep = [c for c in har_names if any(f"{a}_ma_" in c for a in adj_cols)]
    d2 = d2.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = d2[keep].to_numpy(dtype=np.float64)
    ref_iqr, fixed = _build_scale_guards(
        X,
        d2,
        keep,
        erode_ma_window=guard_fix,
        warmup_floor=1.0 if guard_fix else 0.0,
    )
    Z = rolling_robust_scale(
        X, TRAIN_WINDOW_DAYS * PERIODS_PER_DAY, ref_iqr=ref_iqr, fixed_cols=fixed
    )
    return Z, keep, d2


def stage_fix() -> None:
    """Add one fix at a time and watch the invariant, never a score.

    ``diagnose`` + ``trace`` established the mechanisms; this measures whether each fix removes the
    one it targets. Four fixes, in the order they were derived:

    1. ``band`` — the two-sided per-slot band on the diurnal baseline
       (``DIURNAL_SLOT_BAND_FRAC``). Targets the eleven ``sum*stock`` / ``stocktwits_sentiment``
       columns, all of which are a within-slot baseline collapse.
    2. ``excl`` — ``DIURNAL_EXCLUDED`` matched as stems, so its ``vix`` and ``sentiment`` entries
       cover ``vvix`` / ``vix3m`` / ``stocktwits_sentiment`` as the constant's name says they should.
    3. ``mode`` — the hurdle encoding keyed on the modal value rather than on zero. Targets
       ``numobs``, which is 30 on 78% of bars.
    4. ``guard`` — the scale guard's availability mask eroded by the MA window, plus a nonzero
       reference during the unmeasurable warm-up. Targets ``adj_vix3m_ma_3125`` /
       ``adj_vvix_ma_3125`` (the imputation seam) and the ``ref_iqr == 0`` rows.

    The band fraction is the one free parameter, and it is chosen here by which value clears the
    invariant with the least intervention — *not* by a forecast score. That distinction is the whole
    point of §10: an invariant consumes no inferential budget because it never looks at accuracy.
    """
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for label, band, excl, mode_h, guard in ARMS:
        Z, names, _ = _build_flagged(band, excl, mode_h, guard)
        a = np.abs(Z)
        worst = int(np.nanargmax(a.max(0)))
        per_col = a.max(0)
        rows.append(
            {
                "arm": label,
                "slot_band": band,
                "stem_exclusion": excl,
                "mode_hurdle": mode_h,
                "guard_fix": guard,
                "max_abs_z": float(per_col[worst]),
                "worst_col": names[worst],
                "n_cols_over_50": int((per_col >= 50).sum()),
                "n_cols_over_20": int((per_col >= 20).sum()),
                "n_cols": len(names),
                "p99.9_panelwide": float(np.nanpercentile(a, 99.9)),
                "n_rows_over_20": int((a >= 20).sum()),
            }
        )
        print(
            f"  {label:16s} max|z| {rows[-1]['max_abs_z']:7.1f} ({rows[-1]['worst_col']})  "
            f"cols>=50 {rows[-1]['n_cols_over_50']:2d}  cols>=20 {rows[-1]['n_cols_over_20']:2d}"
            f" / {len(names)}  rows>=20 {rows[-1]['n_rows_over_20']:6d}",
            flush=True,
        )
        if label == "pre":
            got = float(per_col[names.index(PRE_FIX_WORST[0])])
            if abs(got - PRE_FIX_WORST[1]) > 0.5:
                raise AssertionError(
                    f"the 'pre' arm no longer reproduces the recorded pre-fix state: "
                    f"{PRE_FIX_WORST[0]} is {got:.1f}, recorded {PRE_FIX_WORST[1]:.1f}. "
                    "Every comparison below it would be against the wrong baseline."
                )
            print(f"    (control: {PRE_FIX_WORST[0]} reproduces at {got:.1f})", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/tailfix_ladder.csv", index=False)

    # Per-column before/after under the full fix, so the effect is attributable per family.
    Zp, names, _ = _build_flagged(*ARMS[0][1:])
    Za, names_a, _ = _build_flagged(*ARMS[-1][1:])
    common = [n for n in names if n in names_a]
    cmp = pd.DataFrame(
        {
            "feature": common,
            "max_abs_z_pre": [float(np.abs(Zp[:, names.index(n)]).max()) for n in common],
            "max_abs_z_post": [float(np.abs(Za[:, names_a.index(n)]).max()) for n in common],
        }
    )
    cmp["ratio"] = cmp["max_abs_z_post"] / cmp["max_abs_z_pre"]
    cmp = cmp.sort_values("max_abs_z_pre", ascending=False)
    print("\nPER-COLUMN, pre vs all-fixes (top 25 by pre)")
    print(cmp.head(25).to_string(index=False), flush=True)
    print(f"\n  columns UNCHANGED (ratio within 1e-6 of 1): "
          f"{int((cmp['ratio'].sub(1).abs() < 1e-6).sum())} / {len(cmp)}")
    cmp.to_csv(f"{OUT}/tailfix_per_column.csv", index=False)
    print(f"wrote {OUT}/tailfix_ladder.csv + tailfix_per_column.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["diagnose", "trace", "modes", "fix"], required=True)
    a = ap.parse_args()
    {
        "diagnose": stage_diagnose,
        "trace": stage_trace,
        "modes": stage_modes,
        "fix": stage_fix,
    }[a.stage]()
